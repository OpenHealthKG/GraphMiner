package net.openhealthkg.graphminer.edgeminer;

import static org.apache.spark.sql.functions.*;

import com.azure.ai.openai.OpenAIClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.ai.openai.models.*;
import com.azure.core.credential.AzureKeyCredential;
import net.openhealthkg.graphminer.Util;
import net.openhealthkg.graphminer.heuristics.PXYHeuristic;
import org.apache.hadoop.shaded.org.checkerframework.checker.units.qual.C;
import org.apache.spark.Aggregator;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import com.azure.ai.openai.OpenAIClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.ai.openai.models.Embeddings;
import com.azure.ai.openai.models.EmbeddingsOptions;
import com.azure.core.credential.AzureKeyCredential;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class EdgeMiner {


    public Dataset<Row> mineEdges(Dataset<Row> df, long cohortSize, int keepTopN, PXYHeuristic... heuristics) {
        cohortSize = cohortSize == 0 ? df.select("occurrence_id").distinct().count() : cohortSize;
        // Get a dataset of node IDs and names for the purposes of node description embeddings
        Dataset<Row> nodeIDsAndDescs = df.select("node_id", "node_description").distinct().persist();
        // Map to integer IDs for space and retain the mappings
        df = df.select("node_id", "occurrence_id").distinct();
        Tuple2<Dataset<Row>, Dataset<Row>> mapped = Util.mapIDstoNumeric(df, "node_id");
        df = mapped._1;
        Dataset<Row> mappings = mapped._2.persist();
        Map<String, String> externalNodeIDtoDescriptionMapping = new HashMap<>();
        long numNodes = mappings.count();
        df = Util.mapIDstoNumeric(df, "occurrence_id")._1; // We don't need to retain the original occurrence_id
        // Perform the actual scoring.
        Dataset<Row> scoreTermPairs = scoreTermPairs(df, cohortSize, heuristics);
        // Filter top N scores
        if (keepTopN > 0) {
            scoreTermPairs = keepTopN(scoreTermPairs, keepTopN);
        }
        scoreTermPairs.persist(StorageLevel.DISK_ONLY()); // Persist this (very large) dataset to disk now that we are not doing any further ops TODO
        Dataset<Row> pcaSimScoring = applyPCAonHeuristics(df, heuristics);
        Dataset<Row> heuristicFeatureVectors = vectorizeHeuristics(scoreTermPairs, numNodes, heuristics);
        Dataset<Row> nodeNameVectors = getTextEmbeddingsForDescription(mappings.withColumnRenamed("src_node_id", "node_id"));

    }

    private Dataset<Row> getTextEmbeddingsForDescription(Dataset<Row> df) {
        int batch_size = 1024;
        StructType schema = new StructType()
                .add("node_id", DataTypes.StringType, false)
                .add("node_embeddings", new VectorUDT(), false);

        return df.mapPartitions(
                (MapPartitionsFunction<Row, Row>) it -> {
                    List<String> node_ids = new ArrayList<>();
                    List<String> batch = new ArrayList<>();
                    List<Row> results = new ArrayList<>();

                    while (it.hasNext()) {
                        Row r = it.next();
                        if (batch.size() >= batch_size) {
                            // Process batch
                            processEmbeddingBatch(node_ids, batch, results);
                            node_ids.clear();
                            batch.clear();
                        }
                        node_ids.add(r.getString(r.fieldIndex("node_id")));
                        batch.add(r.getString(r.fieldIndex("node_description")));
                    }

                    // Process remaining items
                    if (!batch.isEmpty()) {
                        processEmbeddingBatch(node_ids, batch, results);
                    }

                    return results.iterator();
                }, RowEncoder.encoderFor(schema)
        );
    }

    private void processEmbeddingBatch(List<String> node_ids, List<String> texts, List<Row> results) {
        OpenAIClient client = new OpenAIClientBuilder()
                .credential(new AzureKeyCredential(System.getenv("AZURE_OPENAI_KEY")))
                .endpoint(System.getenv("AZURE_OPENAI_ENDPOINT"))
                .buildClient();

            EmbeddingsOptions options = new EmbeddingsOptions(texts);
            Embeddings embeddings = client.getEmbeddings("text-embedding-3-large", options);
            int i = 0;
            for (EmbeddingItem item : embeddings.getData()) {
                String node_id = node_ids.get(i);
                double[] vector = item.getEmbedding().stream().mapToDouble(Float::doubleValue).toArray();
                results.add(RowFactory.create(node_id, Vectors.dense(vector)));
            }
        }
    }

    /**
     * @param df An input data frame consisting of (at a minimum) a string term/node identifier (located in the
     *           node_id column) and a string occurrence (document/patient) identifier.
     * @param cohortSize cohort size
     * @param heuristics the Heuristics to use
     * @return A dataframe consisting of x_node_id, y_node_id, and each heuristic
     */
    public Dataset<Row> scoreTermPairs(Dataset<Row> df, long cohortSize, PXYHeuristic... heuristics) {
        Dataset<Row> nodeFreqs = df.groupBy("node_id").count();

        // Do a cartesian product to get all (x,y) combinations possible against which we build our frequency lists
        Dataset<Row> nodes_x = df.select("node_id").distinct().withColumnRenamed("node_id", "x_node_id");
        Dataset<Row> nodes_y = df.select("node_id").distinct().withColumnRenamed("node_id", "y_node_id");
        Dataset<Row> ret = nodes_x.crossJoin(
                nodes_y
        ).where(
                col("x_node_id").lt(col("y_node_id")) //  prevent (x,y) and (y,x) both showing up
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
        ret = ret.join(dfxy, ret.col("x_node_id").equalTo(dfxy.col("x_node_id").and(ret.col("y_node_id").equalTo(dfxy.col("y_node_id")))), "left").select(
                ret.col("x_node_id"),
                ret.col("y_node_id"),
                ret.col("fx"),
                ret.col("fy"),
                coalesce(dfxy.col("fxy"), lit(0)).alias("fxy")
        ).withColumn("cohort_size", lit(cohortSize));

        // Now calculate heuristics
        for (PXYHeuristic heuristic : heuristics) {
            ret = ret.withColumn(heuristic.getHeuristicName() + "_raw", udf(heuristic, DataTypes.DoubleType).apply(ret.col("fx"), ret.col("fy"), ret.col("fxy"), ret.col("cohort_size")));
        }
        // Now we need to re-scale to 0->1 w/ outlier handling. To do this, we do log scale divided by max log.
        for (PXYHeuristic heuristic : heuristics) {
            String rawCol = heuristic.getHeuristicName() + "_raw";
            String scaledCol = heuristic.getHeuristicName();

            Double maxLog = ret
                    .select(max(log1p(col(rawCol))).alias("max_log"))
                    .first()
                    .getDouble(0);

            if (maxLog == 0.0) {
                ret = ret.withColumn(scaledCol, lit(0.0));
                continue;
            }
            ret = ret.withColumn(
                    scaledCol,
                    when(col(rawCol).equalTo(0), lit(0.0))
                            .otherwise(log1p(col(rawCol)).divide(lit(maxLog)))
            );
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
        df = df.withColumn("sim_score", element_at(callUDF("vector_to_array", col("pca_vec")), 1));
        return df.select("x_node_id", "y_node_id", "sim_score");

    }

    public Dataset<Row> vectorizeHeuristics(Dataset<Row> df, long numNodes, PXYHeuristic... heuristics) {
        return df.repartition(functions.col("x_node_id")).mapPartitions(
                (Iterator<Row> it) -> {
                    List<Integer> indices = new ArrayList<>();
                    List<Double> values = new ArrayList<>();
                    AtomicInteger x_node_id = new AtomicInteger(-1);
                    it.forEachRemaining(r -> {
                        if (x_node_id.get() == -1) {
                            x_node_id.set(Math.toIntExact(r.getLong(r.fieldIndex("x_node_id"))));
                        }
                        Integer offset = Math.toIntExact((r.getLong(r.fieldIndex("y_node_id")) - 1) * heuristics.length); // ID remapping uses row_number() which is 1-indexed
                        int i = 0;
                        for (PXYHeuristic heuristic : heuristics) {
                            indices.add(offset + i);
                            values.add(r.getDouble(r.fieldIndex(heuristic.getHeuristicName())));
                            i++;
                        }
                    });
                    return Collections.singleton(RowFactory.create(x_node_id, new SparseVector(Math.toIntExact(numNodes * heuristics.length), indices.stream().mapToInt(Integer::intValue).toArray(), values.stream().mapToDouble(Double::doubleValue).toArray()))).iterator();
                },
                RowEncoder.encoderFor(
                        new StructType(
                                new StructField[]{
                                        StructField.apply("x_node_id", DataTypes.LongType, false, Metadata.empty()),
                                        StructField.apply("heurisitics_vector", new VectorUDT(), false, Metadata.empty())
                                }
                        )
                )
        );
    }
}
