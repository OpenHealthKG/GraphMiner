package net.openhealthkg.graphminer;

import static org.apache.spark.sql.functions.*;

import net.openhealthkg.graphminer.edgeminer.EdgeMiner;
import net.openhealthkg.graphminer.heuristics.ChiSquared;
import net.openhealthkg.graphminer.heuristics.JLH;
import net.openhealthkg.graphminer.heuristics.MutualInformation;
import net.openhealthkg.graphminer.heuristics.NormalizedGoogleDistance;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

/**
 *
 */
public class FromOMOP {
    public static void main(String... args) {
        SparkSession spark = SparkSession.builder().getOrCreate();
        String tag = spark.conf().get("spark.openhealthkg.run_tag", "OMOP");
        int pruneThreshold = Integer.valueOf(spark.conf().get("spark.openhealthkg.prune_edges_at_rank", "100"));
        long cohortSize = spark.table("person")
                .select(col("person_id"))
                .distinct()
                .count();

        EdgeMiner miner = new EdgeMiner();
        // condition_occurrence: (person_id, condition_concept_id)
        Dataset<Row> cond = spark.table("condition_occurrence")
                .select(
                        col("condition_concept_id").cast("string").alias("node_id"),
                        col("person_id").cast("string").alias("occurrence_id")
                )
                .where(col("node_id").isNotNull().and(col("node_id").notEqual("0")));

        // drug_exposure: (person_id, drug_concept_id)
        Dataset<Row> drug = spark.table("drug_exposure")
                .select(
                        col("drug_concept_id").cast("string").alias("node_id"),
                        col("person_id").cast("string").alias("occurrence_id")
                )
                .where(col("node_id").isNotNull().and(col("node_id").notEqual("0")));

        // procedure_occurrence: (person_id, procedure_concept_id)
        Dataset<Row> proc = spark.table("procedure_occurrence")
                .select(
                        col("procedure_concept_id").cast("string").alias("node_id"),
                        col("person_id").cast("string").alias("occurrence_id")
                )
                .where(col("node_id").isNotNull().and(col("node_id").notEqual("0")));
        // device_exposure: (person_id, device_concept_id)
        Dataset<Row> device = spark.table("device_exposure")
                .select(
                        col("device_concept_id").cast("string").alias("node_id"),
                        col("person_id").cast("string").alias("occurrence_id")
                )
                .where(col("node_id").isNotNull().and(col("node_id").notEqual("0")));

        // measurement: (person_id, measurement_concept_id)
        Dataset<Row> measurement = spark.table("measurement")
                .select(
                        col("measurement_concept_id").cast("string").alias("node_id"),
                        col("person_id").cast("string").alias("occurrence_id")
                )
                .where(col("node_id").isNotNull().and(col("node_id").notEqual("0")));

        // Union into node_id, occurrence_id format
        Dataset<Row> nodeOccurrences = cond.unionByName(drug).unionByName(proc).unionByName(device).unionByName(measurement).distinct().withColumn("tag", lit(tag)).filter(col("node_id").isNotNull().and(col("node_id").notEqual("0")));

        // Load concept relationships
        Dataset<Row> relationships = spark.table("concept_relationship")
                .select(
                        col("concept_id_1").cast("string").alias("src_node_id"),
                        col("concept_id_2").cast("string").alias("tgt_node_id"),
                        col("relationship_id").alias("edge_label")
                );

        // Generate edge labels from concept relationships
        Dataset<Row> edgeLabels = relationships
                .join(nodeOccurrences.select("node_id").distinct().alias("source"),
                        col("src_node_id").equalTo(col("source.node_id")))
                .join(nodeOccurrences.select("node_id").distinct().alias("target"),
                        col("tgt_node_id").equalTo(col("target.node_id")))
                .select("src_node_id", "tgt_node_id", "edge_label")
                .distinct();

        // Save edge labels
        edgeLabels.write().mode("overwrite").parquet("openhealthkg_data/raw/" + tag + "/labels");

        // Generate raw features
        miner.generateEdgeFeatures(
                spark,
                "openhealthkg_data/raw/" + tag,
                tag,
                nodeOccurrences,
                edgeLabels,
                cohortSize,
                pruneThreshold,
                new MutualInformation(), new ChiSquared(), new NormalizedGoogleDistance(), new JLH());

    }
    
    


}
