import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from pyarrow import parquet as pq
import numpy as np
from torch import nn
from torch.nn import functional as F

ID_COLS = ["tag", "x_node_id", "y_node_id", "x_node_type", "y_node_type"]
SCALAR_COLS = ["cos_sim", "euclidean_distance", "dot_product", "manhattan_distance", "sim_score"]
LABEL_COL = "edge_label"
X_EMB_COL = "x_node_embeddings"
Y_EMB_COL = "y_node_embeddings"
X_TYPE_COL = "x_node_type"
Y_TYPE_COL = "y_node_type"

@dataclass
class TrainConfig:
    parquet_glob: str
    num_edge_types: int

    # PU knobs
    unlabeled_keep_prob: float = 0.2  # downsample label==0
    unlabeled_weight: float = 0.1  # BCE weight for unlabeled rows
    type_loss_weight: float = 0.5  # lambda for multiclass head

    # Training knobs
    batch_size: int = 2048
    num_workers: int = 2
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Model dims
    emb_proj_dim: int = 128
    type_emb_dim: int = 16
    hidden: int = 256
    dropout: float = 0.1

class EdgeParquetIterable(torch.utils.data.IterableDataset):
    """
    Streams parquet shards written by your Spark job.
    Expects:
      - x_node_embeddings, y_node_embeddings as array<double>
      - scalar cols as doubles
      - x_node_type, y_node_type as ints
      - edge_label int (0..K)
    """
    def __init__(self, files: List[str], cfg: TrainConfig, shuffle_files=True):
        super().__init__()
        self.files = files
        self.cfg = cfg
        self.shuffle_files = shuffle_files

    def _iter_file(self, path: str):
        pf = pq.ParquetFile(path)
        cols = ID_COLS + SCALAR_COLS + [X_EMB_COL, Y_EMB_COL, LABEL_COL]
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=cols).to_pydict()

            y = np.asarray(table[LABEL_COL], dtype=np.int64)

            # PU downsampling of unlabeled
            if self.cfg.unlabeled_keep_prob < 1.0:
                keep = np.ones_like(y, dtype=bool)
                unl = (y == 0)
                if unl.any():
                    keep_unl = (np.random.rand(unl.sum()) < self.cfg.unlabeled_keep_prob)
                    keep[unl] = keep_unl
            else:
                keep = slice(None)

            # Scalars
            Xs = np.stack([np.asarray(table[c], dtype=np.float32) for c in SCALAR_COLS], axis=1)[keep]

            # Types
            xt = np.asarray(table[X_TYPE_COL], dtype=np.int64)[keep]
            yt = np.asarray(table[Y_TYPE_COL], dtype=np.int64)[keep]

            # Embeddings: list-of-lists -> array
            # (assumes all rows have same embedding length)
            xemb = np.asarray(table[X_EMB_COL], dtype=np.float32)[keep]
            yemb = np.asarray(table[Y_EMB_COL], dtype=np.float32)[keep]

            yy = y[keep]

            for i in range(len(yy)):
                yield Xs[i], xemb[i], yemb[i], xt[i], yt[i], int(yy[i])

    def __iter__(self):
        # DDP sharding by rank + dataloader workers
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))

        files = list(self.files)
        if self.shuffle_files:
            random.shuffle(files)

        files = files[rank::world]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid = worker_info.id
            wnum = worker_info.num_workers
            files = files[wid::wnum]

        for f in files:
            yield from self._iter_file(f)


def collate(batch):
    Xs, xemb, yemb, xt, yt, y = zip(*batch)
    return (
        torch.tensor(np.stack(Xs), dtype=torch.float32),
        torch.tensor(np.stack(xemb), dtype=torch.zfloat32),
        torch.tensor(np.stack(yemb), dtype=torch.float32),
        torch.tensor(xt, dtype=torch.long),
        torch.tensor(yt, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


class TypedPUEdgeModel(nn.Module):
    # Use a three-tower approach to prevent early cross-contamination of features (particularly due to large
    # embedding sizes for text/node desc embeddings)

    def __init__(self, emb_dim: int, num_types: int, cfg: TrainConfig, max_type_id: int = 1024):
        super().__init__()
        self.cfg = cfg

        # Type embeddings
        self.type_emb = nn.Embedding(max_type_id + 1, cfg.type_emb_dim)

        # Scalars tower
        self.scalar_norm = nn.LayerNorm(len(SCALAR_COLS))
        self.scalar_tower = nn.Sequential(
            nn.Linear(len(SCALAR_COLS), cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )

        # Embeddings tower: project x and y, then interactions
        self.x_proj = nn.Linear(emb_dim, cfg.emb_proj_dim, bias=False)
        self.y_proj = nn.Linear(emb_dim, cfg.emb_proj_dim, bias=False)
        self.emb_norm = nn.LayerNorm(cfg.emb_proj_dim)

        # Input dim to emb tower = [x, y, x*y, |x-y|, dot] where dot is 1
        emb_in = cfg.emb_proj_dim * 4 + 1
        self.emb_tower = nn.Sequential(
            nn.Linear(emb_in, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )

        # Types tower
        type_in = cfg.type_emb_dim * 2
        self.type_tower = nn.Sequential(
            nn.Linear(type_in, cfg.hidden),
            nn.ReLU(),
        )

        # Gated fusion: learn weights for each tower output
        self.gate = nn.Linear(cfg.hidden * 3, 3)

        # Shared trunk after fusion
        self.shared = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Heads
        self.edge_head = nn.Linear(cfg.hidden, 1)        # binary logit
        self.type_head = nn.Linear(cfg.hidden, num_types) # K logits (for labels 1..K)

    def forward(self, scalars, xemb, yemb, xt, yt):
        # Scalars
        s = self.scalar_tower(self.scalar_norm(scalars))

        # Types
        tx = self.type_emb(xt.clamp_min(0))
        ty = self.type_emb(yt.clamp_min(0))
        t = self.type_tower(torch.cat([tx, ty], dim=1))

        # Embeddings
        x = self.emb_norm(self.x_proj(xemb))
        y = self.emb_norm(self.y_proj(yemb))
        had = x * y
        abd = torch.abs(x - y)
        dot = torch.sum(x * y, dim=1, keepdim=True)
        e = self.emb_tower(torch.cat([x, y, had, abd, dot], dim=1))

        # Gate & fuse
        cat = torch.cat([s, e, t], dim=1)
        g = F.softmax(self.gate(cat), dim=1)  # (B,3)
        h = g[:, 0:1] * s + g[:, 1:2] * e + g[:, 2:3] * t

        h = self.shared(h)
        edge_logit = self.edge_head(h).squeeze(1)
        type_logits = self.type_head(h)
        return edge_logit, type_logits, g.detach()

def pu_typed_loss(edge_logit, type_logits, y, unlabeled_weight: float, type_loss_weight: float,
                  class_weights: Optional[torch.Tensor] = None):
    y_bin = (y > 0).float()

    # Weighted BCE: downweight unlabeled rows
    w = torch.ones_like(y_bin)
    w[y == 0] = unlabeled_weight
    bce = F.binary_cross_entropy_with_logits(edge_logit, y_bin, weight=w)

    # Type CE only on labeled positives
    pos = (y > 0)
    if pos.any():
        y_type = (y[pos] - 1).long()
        ce = F.cross_entropy(type_logits[pos], y_type, weight=class_weights)
    else:
        ce = torch.tensor(0.0, device=y.device)

    loss = bce + type_loss_weight * ce
    return loss, bce.detach(), ce.detach()


def setup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0

def cleanup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.destroy_process_group()

def train(cfg: TrainConfig):
    files = sorted(glob.glob(cfg.parquet_glob))
    assert files, f"No parquet matched: {cfg.parquet_glob}"

    ddp, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    pf0 = pq.ParquetFile(files[0])
    sample = pf0.read_row_group(0, columns=[X_EMB_COL]).to_pydict()[X_EMB_COL][0]
    emb_dim = len(sample)

    ds = EdgeParquetIterable(files, cfg, shuffle_files=True)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    model = TypedPUEdgeModel(emb_dim=emb_dim, num_types=cfg.num_edge_types, cfg=cfg, max_type_id=600).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    class_weights = None

    for epoch in range(cfg.epochs):
        model.train()
        for step, (scalars, xemb, yemb, xt, yt, y) in enumerate(dl):
            scalars = scalars.to(device, non_blocking=True)
            xemb = xemb.to(device, non_blocking=True)
            yemb = yemb.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                edge_logit, type_logits, gate = model(scalars, xemb, yemb, xt, yt)
                loss, bce, ce = pu_typed_loss(
                    edge_logit, type_logits, y,
                    unlabeled_weight=cfg.unlabeled_weight,
                    type_loss_weight=cfg.type_loss_weight,
                    class_weights=class_weights
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % 50 == 0 and int(os.environ.get("RANK", "0")) == 0:
                gmean = gate.mean(dim=0).cpu().numpy()
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"bce={bce.item():.4f} ce={ce.item():.4f} "
                    f"gate=[scalar={gmean[0]:.2f}, emb={gmean[1]:.2f}, type={gmean[2]:.2f}]"
                )

    cleanup_ddp()
    return model


if __name__ == "__main__":
    cfg = TrainConfig(
        parquet_glob="/path/to/full_dataset_vectors/*.parquet",
        num_edge_types=5,
        unlabeled_keep_prob=0.2,
        unlabeled_weight=0.1,
        type_loss_weight=0.5,
        batch_size=2048,
        epochs=3,
    )
    train(cfg)