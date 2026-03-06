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
    # Partitioned input root: expects /partitions/{idx}/*.parquet
    partitions_root: str

    # CV setup
    cv_partitions: List[int] = None          # defaults to [0,1,2,3,4]
    holdout_partition: int = 5               # final test fold

    # Allow auto-infer from data if omitted
    num_edge_types: Optional[int] = None

    # PU knobs
    unlabeled_keep_prob: float = 0.2
    unlabeled_weight: float = 0.1
    type_loss_weight: float = 0.5

    # Training knobs
    batch_size: int = 2048
    num_workers: int = 2
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Model dims
    emb_proj_dim: int = 128
    type_emb_dim: int = 16
    hidden: int = 256
    dropout: float = 0.1

    # Early stopping
    early_stop: bool = True
    patience: int = 3
    min_delta: float = 1e-4
    val_max_batches: int = 200

    def __post_init__(self):
        if self.cv_partitions is None:
            self.cv_partitions = [0, 1, 2, 3, 4]

def get_partition_files(partitions_root: str, partition_idx: int) -> List[str]:
    part_glob = os.path.join(partitions_root, str(partition_idx), "*.parquet")
    files = sorted(glob.glob(part_glob))
    if not files:
        raise AssertionError(f"No parquet matched partition glob: {part_glob}")
    return files


def get_files_for_partitions(partitions_root: str, partition_indices: List[int]) -> List[str]:
    files = []
    for idx in partition_indices:
        files.extend(get_partition_files(partitions_root, idx))
    if not files:
        raise AssertionError(f"No parquet found for partitions: {partition_indices}")
    return sorted(files)


def build_fold_file_sets(cfg: TrainConfig, val_partition: int):
    train_partitions = [p for p in cfg.cv_partitions if p != val_partition]
    val_partitions = [val_partition]

    train_files = get_files_for_partitions(cfg.partitions_root, train_partitions)
    val_files = get_files_for_partitions(cfg.partitions_root, val_partitions)
    return train_files, val_files


def build_final_train_and_test_sets(cfg: TrainConfig):
    train_files = get_files_for_partitions(cfg.partitions_root, cfg.cv_partitions)
    test_files = get_files_for_partitions(cfg.partitions_root, [cfg.holdout_partition])
    return train_files, test_files

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
        torch.tensor(np.stack(xemb), dtype=torch.float32),
        torch.tensor(np.stack(yemb), dtype=torch.float32),
        torch.tensor(xt, dtype=torch.long),
        torch.tensor(yt, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )

def infer_num_edge_types(files):
    max_label = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            y = np.asarray(
                pf.read_row_group(rg, columns=[LABEL_COL]).to_pydict()[LABEL_COL],
                dtype=np.int64
            )
            if y.size:
                ml = int(y.max())
                if ml > max_label:
                    max_label = ml
    return max_label

def evaluate(model, dl, device, cfg: TrainConfig):
    model.eval()
    losses, bces, ces = [], [], []

    for i, (scalars, xemb, yemb, xt, yt, y) in enumerate(dl):
        scalars = scalars.to(device, non_blocking=True)
        xemb = xemb.to(device, non_blocking=True)
        yemb = yemb.to(device, non_blocking=True)
        xt = xt.to(device, non_blocking=True)
        yt = yt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        edge_logit, type_logits, _ = model(scalars, xemb, yemb, xt, yt)
        loss, bce, ce = pu_typed_loss(
            edge_logit, type_logits, y,
            unlabeled_weight=cfg.unlabeled_weight,
            type_loss_weight=cfg.type_loss_weight,
            class_weights=None
        )
        losses.append(float(loss.item()))
        bces.append(float(bce.item()))
        ces.append(float(ce.item()))

        if cfg.val_max_batches and (i + 1) >= cfg.val_max_batches:
            break

    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "bce": float(np.mean(bces)) if bces else float("inf"),
        "ce": float(np.mean(ces)) if ces else float("inf"),
    }

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
    train_files = sorted(glob.glob(cfg.train_glob))
    val_files   = sorted(glob.glob(cfg.val_glob))
    test_files  = sorted(glob.glob(cfg.test_glob))
    assert train_files, f"No parquet matched train_glob: {cfg.train_glob}"
    assert val_files,   f"No parquet matched val_glob:   {cfg.val_glob}"
    assert test_files,  f"No parquet matched test_glob:  {cfg.test_glob}"

    # infer K if needed (scan train+val; test is optional)
    if cfg.num_edge_types is None:
        if int(os.environ.get("RANK", "0")) == 0:
            print("Inferring num_edge_types from labels in train+val...")
        cfg.num_edge_types = infer_num_edge_types(train_files + val_files)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Inferred num_edge_types={cfg.num_edge_types}")

    ddp, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # infer embedding dim from first train file
    pf0 = pq.ParquetFile(train_files[0])
    sample = pf0.read_row_group(0, columns=[X_EMB_COL]).to_pydict()[X_EMB_COL][0]
    emb_dim = len(sample)

    train_ds = EdgeParquetIterable(train_files, cfg, shuffle_files=True)
    val_ds   = EdgeParquetIterable(val_files,   cfg, shuffle_files=False)
    test_ds  = EdgeParquetIterable(test_files,  cfg, shuffle_files=False)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=True, collate_fn=collate
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=True, collate_fn=collate
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=True, collate_fn=collate
    )

    model = TypedPUEdgeModel(emb_dim=emb_dim, num_types=cfg.num_edge_types, cfg=cfg, max_type_id=600).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for step, (scalars, xemb, yemb, xt, yt, y) in enumerate(train_dl):
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
                    class_weights=None
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

        # ---- validate + early stop on rank 0 ----
        if int(os.environ.get("RANK", "0")) == 0:
            metrics = evaluate(model, val_dl, device, cfg)
            print(f"[val] epoch={epoch} loss={metrics['loss']:.4f} bce={metrics['bce']:.4f} ce={metrics['ce']:.4f}")

            improved = (best_val - metrics["loss"]) > cfg.min_delta
            if improved:
                best_val = metrics["loss"]
                bad_epochs = 0
                m = model.module if hasattr(model, "module") else model
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad_epochs += 1
                if cfg.early_stop and bad_epochs >= cfg.patience:
                    print(f"Early stopping at epoch={epoch} (best_val={best_val:.4f})")
                    break

    # restore best before test
    if best_state is not None:
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(best_state)

    if int(os.environ.get("RANK", "0")) == 0:
        t = evaluate(model, test_dl, device, cfg)
        print(f"[test] loss={t['loss']:.4f} bce={t['bce']:.4f} ce={t['ce']:.4f}")

    cleanup_ddp()
    return model


if __name__ == "__main__":
    cfg = TrainConfig(
        train_glob="/path/to/train_dataset_vectors/*.parquet",
        test_glob="/path/to/test_dataset_vectors/*.parquet",
        val_glob="/path/to/val_dataset_vectors/*.parquet",
        num_edge_types=5,
        unlabeled_keep_prob=0.2,
        unlabeled_weight=0.1,
        type_loss_weight=0.5,
        batch_size=2048,
        epochs=3,
    )
    train(cfg)