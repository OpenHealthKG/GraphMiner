package net.openhealthkg.graphminer.heuristics;

import org.apache.spark.sql.api.java.UDF4;

public interface PXYHeuristic extends UDF4<Long, Long, Long, Long, Double> {
    String getHeuristicName();
    @Override
    Double call(Long fx, Long fy, Long fxy, Long coll_size) throws Exception;
}
