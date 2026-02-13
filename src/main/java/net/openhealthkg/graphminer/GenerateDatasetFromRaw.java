package net.openhealthkg.graphminer;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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


    }
}
