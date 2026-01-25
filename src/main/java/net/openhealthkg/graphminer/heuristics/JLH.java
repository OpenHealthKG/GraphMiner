package net.openhealthkg.graphminer.heuristics;

/**
 * Implements the JLH significant terms scorer, as defined by Elasticsearch
 */
public class JLH implements PXYHeuristic {
    @Override
    public String getHeuristicName() {
        return "jlh";
    }

    /**
     * (P(x AND y) - P(x))*(P(x AND y)/P(x)
     */
    @Override
    public Double call(Long fx, Long fy, Long fxy, Long coll_size) throws Exception {
        double pXY = (double)fxy / (double)coll_size;
        double pX = (double)fx / (double)coll_size;
        return (pXY - pX) * (pXY/pX);
    }

}
