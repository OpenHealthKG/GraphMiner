package net.openhealthkg.graphminer;

import org.apache.spark.ml.linalg.BLAS;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class GenerateDatasetFromRaw {
    public static void main(String... args) throws Exception {
        SparkSession spark = SparkSession.builder().getOrCreate();
        String tag = spark.conf().get("spark.openhealthkg.run_tag", "UTP_MHH_OMOP");
        String persistence = spark.conf().get("spark.openhealthkg.persistence", "/data/awen2/projects/OpenHealthKG/openhealthkg_data/");
        String raw = persistence + "/raw/" + tag;
        String out = persistence + "/featurized_datasets/" + tag;
        Dataset<Row> pairs = spark.read().parquet(raw + "/scored_term_pairs").select("x_node_id", "y_node_id").distinct();
        // Construct similarity score metrics for node description embeddings
        // - First load embedding vectors and map to internal vectors
        Dataset<Row> embeddings = spark.read().parquet(raw + "/node_desc_embeddings");
        Dataset<Row> mappings = spark.read().parquet(raw + "/source_node_id_to_vector_index");
        embeddings = embeddings.join(mappings, embeddings.col("node_id").equalTo(mappings.col("tgt_node_id"))).select(mappings.col("tgt_node_id").alias("node_id"), embeddings.col("node_embeddings"));
        // - Now join against pairs to get embeddings for each
        Dataset<Row> pairsWithEmbeddings = pairs
                .join(embeddings, pairs.col("x_node_id").equalTo(embeddings.col("node_id")))
                .join(embeddings.as("y_embeddings"), pairs.col("y_node_id").equalTo(col("y_embeddings.node_id")))
                .select(
                        pairs.col("x_node_id"),
                        pairs.col("y_node_id"),
                        embeddings.col("node_embeddings").alias("x_node_embedding"),
                        col("y_embeddings.node_embeddings").alias("y_node_embedding")
                );
        // - Now calculate vector distance
        Dataset<Row> distances = pairsWithEmbeddings.select(
                col("x_node_id"),
                col("y_node_id"),
                // Cosine similarity
                functions.udf((UDF2<Vector, Vector, Double>) (v1, v2) -> {
                    if (v1 == null || v2 == null) return null;
                    if (v1.size() != v2.size()) throw new IllegalArgumentException("Vector size mismatch");
                    double denom = Vectors.norm(v1, 2.0) * Vectors.norm(v2, 2.0);
                    if (denom == 0.0) return null; // undefined if either vector is zero
                    return BLAS.dot(v1, v2) / denom;
                }, DataTypes.DoubleType).apply(col("x_node_embedding"), col("y_node_embedding")).as("cos_sim"),
                // Euclidean distance
                functions.udf(
                        (UDF2<Vector, Vector, Double>) (v1, v2) -> {
                            if (v1 == null || v2 == null) return null;
                            if (v1.size() != v2.size()) throw new IllegalArgumentException("Vector size mismatch");

                            double[] aa = v1.toArray();
                            double[] bb = v2.toArray();

                            double sumSq = 0.0;
                            for (int i = 0; i < aa.length; i++) {
                                double d = aa[i] - bb[i];
                                sumSq += d * d;
                            }
                            return Math.sqrt(sumSq);
                        }, DataTypes.DoubleType
                ).apply(col("x_node_embedding"), col("y_node_embedding")).as("euclidean_distance"),
                // Dot product
                functions.udf((UDF2<Vector, Vector, Double>) BLAS::dot, DataTypes.DoubleType).apply(col("x_node_embedding"), col("y_node_embedding")).as("dot_product"),
                // Manhattan distance
                functions.udf((UDF2<Vector, Vector, Double>) (v1, v2) -> {
                    if (v1 == null || v2 == null) return null;
                    if (v1.size() != v2.size()) throw new IllegalArgumentException("Vector size mismatch");

                    double[] aa = v1.toArray();
                    double[] bb = v2.toArray();

                    double sum = 0.0;
                    for (int i = 0; i < aa.length; i++) sum += Math.abs(aa[i] - bb[i]);
                    return sum;
                }, DataTypes.DoubleType).apply(col("x_node_embedding"), col("y_node_embedding")).as("manhattan_distance")
        );
        distances.write().mode("overwrite").parquet(out + "/vector_distances");
        distances = spark.read().parquet(out + "/vector_distances");
        Dataset<Row> heuristics = spark.read().parquet(raw + "/heuristic_feature_vectors");
        Dataset<Row> pcaSimScoring = spark.read().parquet(raw + "/pca_sim_scores");
        // Create feature vectors for x, y pairs and write
        Dataset<Row> df = pairs.join(distances, pairs.col("x_node_id").equalTo(distances.col("x_node_id")).and(pairs.col("y_node_id").equalTo(distances.col("y_node_id")))).select(
                pairs.col("x_node_id"),
                pairs.col("y_node_id"),
                distances.col("cos_sim"),
                distances.col("euclidean_distance"),
                distances.col("dot_product"),
                distances.col("manhattan_distance")
        ).join(
                heuristics,
                pairs.col("x_node_id").equalTo(heuristics.col("x_node_id"))
        ).select(
                pairs.col("x_node_id"),
                pairs.col("y_node_id"),
                distances.col("cos_sim"),
                distances.col("euclidean_distance"),
                distances.col("dot_product"),
                distances.col("manhattan_distance"),
                heuristics.col("heuristics_vector")
        ).join(
                pcaSimScoring,
                pairs.col("x_node_id").equalTo(pcaSimScoring.col("x_node_id")).and(pairs.col("y_node_id").equalTo(pcaSimScoring.col("y_node_id")))
        );
        df.join(
                mappings.select(col("tgt_node_id"), col("src_node_id").alias("x_source_node_id")),
                df.col("x_node_id").equalTo(col("tgt_node_id"))
        ).join(
                mappings.select(col("tgt_node_id"), col("src_node_id").alias("y_source_node_id")),
                df.col("y_node_id").equalTo(col("tgt_node_id"))
        ).select(
                lit(tag).alias("tag"),
                col("x_source_node_id").alias("x_node_id"),
                col("y_source_node_id").alias("y_node_id"),
                distances.col("cos_sim"),
                distances.col("euclidean_distance"),
                distances.col("dot_product"),
                distances.col("manhattan_distance"),
                pcaSimScoring.col("sim_score"),
                heuristics.col("heuristics_vector")
        );
        
    }
    
}
