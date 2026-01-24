package net.openhealthkg.graphminer.edgeminer;

import static org.apache.spark.sql.functions.*;

import net.openhealthkg.graphminer.heuristics.PXYHeuristic;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;

public abstract class EdgeMiner {

    private final PXYHeuristic[] heuristics;

    public EdgeMiner(PXYHeuristic... heuristics) {
        this.heuristics = heuristics;
    }

    /**
     * @param df An input data frame consisting of (at a minimum) a string term/node identifier (located in the
     *           node_id column) and a string occurrence (document/patient) identifier.
     * @param cohortSize cohort size, if known. Otherwise 0 to infer from occurrence_ids
     * @return
     */
    private Dataset<Row> scoreTermPairs(Dataset<Row> df, long cohortSize) {
        df = df.select("node_id", "occurrence_id").distinct();
        if (cohortSize == 0) cohortSize = df.select("occurrence_id").distinct().count();
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
                .where(ox.col("x_node_id").lt(oy.col("y_node_id"))) //  prevent (x,y) and (y,x) both showing up
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
            ret = ret.withColumn(heuristic.getHeuristicName(), udf(heuristic, DataTypes.DoubleType).apply(ret.col("fx"), ret.col("fy"), ret.col("fxy"), ret.col("cohort_size")));
        }
        return ret;
    }
}
