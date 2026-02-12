package net.openhealthkg.graphminer.edgeminer;

import static org.apache.spark.sql.functions.*;

import com.azure.ai.openai.OpenAIClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.ai.openai.models.EmbeddingItem;
import com.azure.core.credential.AzureKeyCredential;
import net.openhealthkg.graphminer.Util;
import net.openhealthkg.graphminer.heuristics.PXYHeuristic;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapGroupsFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import com.azure.ai.openai.models.Embeddings;
import com.azure.ai.openai.models.EmbeddingsOptions;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class EdgeMiner {
    public void generateEdgeFeatures(SparkSession spark, String persistenceDir, String tag, Dataset<Row> df, long cohortSize, int keepTopN, PXYHeuristic... heuristics) {
        spark.udf().register("vec_first", (UDF1<Vector, Double>) v ->
                        (v == null || v.size() == 0) ? null : v.apply(0),
                DataTypes.DoubleType
        );
        cohortSize = cohortSize == 0 ? df.select("occurrence_id").distinct().count() : cohortSize;
        // Get a dataset of node IDs and names for the purposes of node description embeddings
        df.select("tag", "node_id", "node_type", "node_description").distinct().write().parquet(persistenceDir + "/" + "node_metadata");
        Dataset<Row> nodeNameVectors = getTextEmbeddingsForDescription(df.select("node_id",  "node_description").distinct().repartition(2));
        nodeNameVectors.write().parquet(persistenceDir + "/" + "node_desc_embeddings");
        nodeNameVectors = spark.read().parquet(persistenceDir + "/" + "node_desc_embeddings");
        // Map to integer IDs for space and retain the mappings
        df = df.select("node_id", "occurrence_id").distinct();
        Tuple2<Dataset<Row>, Dataset<Row>> mapped = Util.mapIDstoNumeric(df, "node_id");
        df = mapped._1;
        Dataset<Row> mappings = mapped._2;
        mappings.write().parquet(persistenceDir + "/" + "source_node_id_to_vector_index");
        mappings = spark.read().parquet(persistenceDir + "/" + "source_node_id_to_vector_index");
        long numNodes = mappings.count();
        df = Util.mapIDstoNumeric(df, "occurrence_id")._1; // We don't need to retain the original occurrence_id
        // Perform the actual scoring.
        Dataset<Row> scoreTermPairs = scoreTermPairs(df, cohortSize, heuristics);
        // Filter top N scores
        if (keepTopN > 0) {
            scoreTermPairs = keepTopN(scoreTermPairs, keepTopN, heuristics);
        }
        scoreTermPairs.write().parquet(persistenceDir + "/" + "scored_node_pairs");
        scoreTermPairs = spark.read().parquet(persistenceDir + "/" + "scored_node_pairs");
        Dataset<Row> pcaSimScoring = applyPCAonHeuristics(scoreTermPairs, heuristics);
        pcaSimScoring.write().parquet(persistenceDir + "/" + "pca_sim_scores");
        pcaSimScoring = spark.read().parquet(persistenceDir + "/" + "pca_sim_scores");
        Dataset<Row> heuristicFeatureVectors = vectorizeHeuristics(scoreTermPairs, numNodes, heuristics);
        heuristicFeatureVectors.write().parquet(persistenceDir + "/" + "heuristic_feature_vectors");
    }

    private static void processEmbeddingBatch(
            OpenAIClient client,
            List<String> nodeIds,
            List<String> texts,
            List<Row> out
    ) {
        EmbeddingsOptions options = new EmbeddingsOptions(texts);
        Embeddings embeddings = client.getEmbeddings("text-embedding-3-large", options);

        int i = 0;
        for (EmbeddingItem item : embeddings.getData()) {
            String nodeId = nodeIds.get(i++);
            double[] vector = item.getEmbedding().stream().mapToDouble(Float::doubleValue).toArray();
            out.add(RowFactory.create(nodeId, Vectors.dense(vector)));
        }
    }

    private Dataset<Row> getTextEmbeddingsForDescription(Dataset<Row> df) {
        int BATCH_SIZE = 256;
        StructType schema = new StructType()
                .add("node_id", DataTypes.StringType, false)
                .add("node_embeddings", new VectorUDT(), false);

        return df.mapPartitions((MapPartitionsFunction<Row, Row>) it -> {
            // executor-side; created once per partition
            OpenAIClient client = new OpenAIClientBuilder()
                    .credential(new AzureKeyCredential(System.getenv("AZURE_OPENAI_KEY")))
                    .endpoint(System.getenv("AZURE_OPENAI_ENDPOINT"))
                    .buildClient();

            List<Row> out = new ArrayList<>();
            List<String> nodeIds = new ArrayList<>(BATCH_SIZE);
            List<String> texts = new ArrayList<>(BATCH_SIZE);

            while (it.hasNext()) {
                Row r = it.next();
                nodeIds.add(r.getString(r.fieldIndex("node_id")));
                texts.add(r.getString(r.fieldIndex("node_description")));

                if (nodeIds.size() >= BATCH_SIZE) {
                    processEmbeddingBatch(client, nodeIds, texts, out);
                    nodeIds.clear();
                    texts.clear();
                }
            }

            if (!nodeIds.isEmpty()) {
                processEmbeddingBatch(client, nodeIds, texts, out);
            }

            return out.iterator();
        }, RowEncoder.encoderFor(schema));
    }


    /**
     * @param df An input data frame consisting of (at a minimum) a string term/node identifier (located in the
     *           node_id column) and a string occurrence (document/patient) identifier.
     * @param cohortSize cohort size
     * @param heuristics the Heuristics to use
     * @return A dataframe consisting of x_node_id, y_node_id, and each heuristic
     */
    public Dataset<Row> scoreTermPairs(Dataset<Row> df, long cohortSize, PXYHeuristic... heuristics) {
        Dataset<Row> nodeFreqs = df.groupBy("node_id").count().filter(col("count").geq(functions.lit(10))); // For de-identification safety and prevent rare from dominating correlations

        // Do a cartesian product to get all (x,y) combinations possible against which we build our frequency lists
        Dataset<Row> nodes_x = nodeFreqs.select("node_id").withColumnRenamed("node_id", "x_node_id");
        Dataset<Row> nodes_y = nodeFreqs.select("node_id").withColumnRenamed("node_id", "y_node_id");
        Dataset<Row> ret = nodes_x.crossJoin(
                nodes_y
        ).select("x_node_id", "y_node_id");

        // Get x, y freqs
        Dataset<Row> dfx = nodeFreqs.withColumnRenamed("node_id", "x_node_id").withColumnRenamed("count", "fx");
        Dataset<Row> dfy = nodeFreqs.withColumnRenamed("node_id", "y_node_id").withColumnRenamed("count", "fy");

        // Do a join to get (x ^ y) freqs
        Dataset<Row> ox = df.withColumnRenamed("node_id", "x_node_id");
        Dataset<Row> oy = df.withColumnRenamed("node_id", "y_node_id");
        Dataset<Row> dfxy = ox.join(oy, ox.col("occurrence_id").equalTo(oy.col("occurrence_id")))
                .select("x_node_id", "y_node_id")
                .groupBy("x_node_id", "y_node_id")
                .agg(count(lit(1)).as("fxy"));

        // Join all frequencies together
        ret = ret.join(dfx, ret.col("x_node_id").equalTo(dfx.col("x_node_id")), "left").select(
                ret.col("x_node_id"),
                ret.col("y_node_id"),
                coalesce(dfx.col("fx"), lit(0)).alias("fx")
        );
        ret = ret.join(dfy, ret.col("y_node_id").equalTo(dfy.col("y_node_id")), "left").select(
                ret.col("x_node_id"),
                ret.col("y_node_id"),
                ret.col("fx"),
                coalesce(dfy.col("fy"), lit(0)).alias("fy")
        );
        ret = ret.join(
                dfxy,
                ret.col("x_node_id").equalTo(dfxy.col("x_node_id")).and(ret.col("y_node_id").equalTo(dfxy.col("y_node_id"))), "left"
        ).select(
                ret.col("x_node_id"),
                ret.col("y_node_id"),
                ret.col("fx"),
                ret.col("fy"),
                coalesce(dfxy.col("fxy"), lit(0)).alias("fxy")
        ).withColumn("cohort_size", lit(cohortSize));

        // Now calculate heuristics
        for (PXYHeuristic heuristic : heuristics) {
            ret = ret.withColumn(
                    heuristic.getHeuristicName() + "_raw",
                    coalesce(udf(heuristic, DataTypes.DoubleType).apply(col("fx"), col("fy"), col("fxy"), col("cohort_size")), lit(0.0))
            );
        }
        // Now we need to re-scale to 0->1 w/ outlier handling. To do this, we do signed log scale divided by signed max log.
        for (PXYHeuristic heuristic : heuristics) {

            String rawCol = heuristic.getHeuristicName() + "_raw";
            String scaledCol = heuristic.getHeuristicName();

            String signedLogCol = rawCol + "_signed_log";

            ret = ret.withColumn(
                    signedLogCol,
                    signum(col(rawCol))
                            .multiply(log1p(abs(col(rawCol))))
            );

            Double maxAbsLog = ret
                    .select(max(abs(col(signedLogCol))).alias("max_abs_log"))
                    .first()
                    .getDouble(0);

            if (maxAbsLog == null || maxAbsLog == 0.0) {
                ret = ret.withColumn(scaledCol, lit(0.5));
            } else {

                // normalize to [-1, 1]
                Column normalized =
                        col(signedLogCol).divide(lit(maxAbsLog));

                // map to [0, 1]
                ret = ret.withColumn(
                        scaledCol,
                        normalized.plus(1.0).divide(2.0)
                );
            }

            ret = ret.drop(signedLogCol);
        }
        List<Column> finalCols = new ArrayList<>();

        finalCols.add(col("x_node_id"));
        finalCols.add(col("y_node_id"));

        for (PXYHeuristic heuristic : heuristics) {
            String name = heuristic.getHeuristicName();
            finalCols.add(col(name));
        }

        return ret.select(finalCols.toArray(new Column[0]));
    }

    public Dataset<Row> keepTopN(Dataset<Row> df, int n, PXYHeuristic... heuristics) {
        return df.withColumn("min_rank", functions.least(
                Arrays.stream(heuristics).map(h -> functions.row_number().over(Window.partitionBy("x_node_id").orderBy(functions.col(h.getHeuristicName()).desc_nulls_last()))).toArray(Column[]::new))
        ).filter(functions.col("min_rank").isNotNull().and(functions.col("min_rank").leq(functions.lit(n)))).drop("min_rank");
    }


    /**
     * Calculates PC1 score for an x/y pair for the purposes of learning a generic sim_score
     * @param df
     * @param heuristics
     * @return
     */
    public Dataset<Row> applyPCAonHeuristics(Dataset<Row> df, PXYHeuristic... heuristics) {

        // logit scale heuristics
        final double eps = 1e-6;
        List<Column> projected = new ArrayList<>();
        projected.add(col("x_node_id"));
        projected.add(col("y_node_id"));
        for (PXYHeuristic heuristic : heuristics) {
            Column clipped = greatest(least(col(heuristic.getHeuristicName()), lit(1.0-eps)), lit(eps));
            Column logit = log(clipped.divide(lit(1.0).minus(clipped))).alias(heuristic.getHeuristicName() + "_logit");
            projected.add(logit);
        }
        df = df.select(projected.toArray(Column[]::new));
        // Setup PCA
        String[] pcaCols = Arrays.stream(heuristics).map(heuristic -> heuristic.getHeuristicName() + "_logit").toArray(String[]::new);
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(pcaCols)
                .setOutputCol("heuristics_vec");
        StandardScaler scaler = new StandardScaler()
                .setInputCol("heuristics_vec")
                .setOutputCol("heuristics_scaled")
                .setWithMean(true)
                .setWithStd(true);
        PCA pca = new PCA()
                .setInputCol("heuristics_scaled")
                .setOutputCol("pca_vec")
                .setK(1);
        // Actually run the PCA
        df = new Pipeline().setStages(new PipelineStage[] {assembler, scaler, pca}).fit(df).transform(df);
        df = df.withColumn("sim_score", callUDF("vec_first", col("pca_vec")));
        return df.select("x_node_id", "y_node_id", "sim_score");

    }

    public Dataset<Row> vectorizeHeuristics(Dataset<Row> df, long numNodes, PXYHeuristic... heuristics) {
        return df.groupByKey(  (MapFunction<Row, Integer>) r -> r.getInt(r.fieldIndex("x_node_id")),
                Encoders.INT()).mapGroups(
                (MapGroupsFunction<Integer, Row, Row>) (xid, it) -> {
                    Map<Integer, Double> valueMap = new HashMap<>();
                    List<Integer> indices = new ArrayList<>();
                    List<Double> values = new ArrayList<>();
                    it.forEachRemaining(r -> {
                        Integer offset = (r.getInt(r.fieldIndex("y_node_id")) - 1) * heuristics.length; // ID remapping uses row_number() which is 1-indexed
                        int i = 0;
                        for (PXYHeuristic heuristic : heuristics) {
                            valueMap.put(offset + i, r.getDouble(r.fieldIndex(heuristic.getHeuristicName())));
                            i++;
                        }
                    });
                    valueMap.keySet().stream().sorted().forEach(k -> {
                        indices.add(k);
                        values.add(valueMap.get(k));
                    });
                    return RowFactory.create(xid, new SparseVector(Math.toIntExact(numNodes * heuristics.length), indices.stream().mapToInt(Integer::intValue).toArray(), values.stream().mapToDouble(Double::doubleValue).toArray()));
                },
                RowEncoder.encoderFor(
                        new StructType(
                                new StructField[]{
                                        StructField.apply("x_node_id", DataTypes.IntegerType, false, Metadata.empty()),
                                        StructField.apply("heuristics_vector", new VectorUDT(), false, Metadata.empty())
                                }
                        )
                )
        );
    }
}
