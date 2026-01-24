package net.openhealthkg.graphminer;

import static org.apache.spark.sql.functions.*;

import net.openhealthkg.graphminer.edgeminer.EdgeMiner;
import net.openhealthkg.graphminer.heuristics.ChiSquared;
import net.openhealthkg.graphminer.heuristics.MutualInformation;
import net.openhealthkg.graphminer.heuristics.NormalizedGoogleDistance;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 */
public class FromOMOP {
    public static void main(String... args) {
        SparkSession spark = SparkSession.builder().getOrCreate();
        long cohortSize = spark.table("person")
                .select(col("person_id"))
                .distinct()
                .count();

        EdgeMiner miner = new EdgeMiner(new ChiSquared(), new MutualInformation(), new NormalizedGoogleDistance());
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

        // Union into node_id, occurrence_id format
        Dataset<Row> nodeOccurrences = cond.unionByName(drug).unionByName(proc).distinct();

        // Now actually get edge scores
        Dataset<Row> df = miner.scoreTermPairs(nodeOccurrences, cohortSize);

        // Write
        df.write().saveAsTable("openhealthkg.edges");

    }


}
