package net.openhealthkg.graphminer.edgeminer;

import static org.apache.spark.sql.functions.*;

import net.openhealthkg.graphminer.Util;
import net.openhealthkg.graphminer.heuristics.PXYHeuristic;
import org.apache.spark.Aggregator;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class EdgeMiner {


    public Dataset<Row> mineEdges(Dataset<Row> df, long cohortSize, int keepTopN, PXYHeuristic... heuristics) {
        cohortSize = cohortSize == 0 ? df.select("occurrence_id").distinct().count() : cohortSize;
        // Map to integer IDs for space and retain the mappings
        df = df.select("node_id", "occurrence_id").distinct();
        Tuple2<Dataset<Row>, Dataset<Row>> mapped = Util.mapIDstoNumeric(df, "node_id");
        df = mapped._1;
        Dataset<Row> mappings = mapped._2.persist();
        long numNodes = mappings.count();
        df = Util.mapIDstoNumeric(df, "occurrence_id")._1; // We don't need to retain the original occurrence_id
        // Perform the actual scoring.
        Dataset<Row> scoreTermPairs = scoreTermPairs(df, cohortSize, heuristics);
        // Filter top N scores
        if (keepTopN > 0) {
            scoreTermPairs = keepTopN(scoreTermPairs, keepTopN);
        }
        scoreTermPairs.persist(StorageLevel.DISK_ONLY()); // Persist this (very large) dataset to disk now that we are not doing any further ops TODO
        Dataset<Row> heuristicFeatureVectors = vectorizeHeuristics(scoreTermPairs, numNodes, heuristics);


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
    };
}
