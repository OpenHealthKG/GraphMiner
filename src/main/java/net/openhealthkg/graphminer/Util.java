package net.openhealthkg.graphminer;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.Arrays;

public class Util {
    /**
     * Compresses potential string IDs and renders suitable for vectorization by mapping to an integer ID
     * @param input
     * @param column
     * @return
     */
    public static Tuple2<Dataset<Row> , Dataset<Row>> mapIDstoNumeric(Dataset<Row> input, String column) {
        String[] columns = input.columns();
        Dataset<Row> mappingDF = input.select(column).distinct().withColumnRenamed(column, "src_" + column).withColumn("tgt_" + column, functions.row_number().over(Window.orderBy(functions.rand(123)))).persist(StorageLevel.DISK_ONLY()); // Persist the mapping DF for consistency
        Dataset<Row> outputDF = input.join(
                mappingDF, input.col(column).equalTo(mappingDF.col("tgt_" + column))
        ).select(
                Arrays.stream(columns).map(s -> s.equals(column) ? functions.col("tgt_" + column).alias(column) : functions.col(column)).toArray(Column[]::new)
        );
        return new Tuple2<>(outputDF, mappingDF);
    }
}
