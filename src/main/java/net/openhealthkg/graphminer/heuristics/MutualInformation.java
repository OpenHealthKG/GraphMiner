package net.openhealthkg.graphminer.heuristics;

/**
 * Implements Mutual Information per Section 13.5.1 eq 13.16 of Manning et al's Introduction to Information Retrieval
 * (2009)
 */
public class MutualInformation implements PXYHeuristic {
    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }

    // Safe version of: p * log2(p/denom)
    private static double safeTerm(double p, double denom) {
        if (p <= 0.0 || denom <= 0.0) return 0.0;
        return p * log2(p / denom);
    }

    @Override
    public Double call(Long fx, Long fy, Long fxy, Long coll_size) throws Exception {
        double pX = (fx / (double)coll_size);
        double pY = (fy / (double)coll_size);
        double pXY = (fxy / (double)coll_size);
        double pNegX = 1 - pX;
        double pNegY = 1 - pY;
        double pXNegY = pX - pXY;
        double pNegXY = pY - pXY;
        double pNegXNegY = 1 - pX - pY + pXY;

        if (pXNegY < 0) pXNegY = 0;
        if (pNegXY < 0) pNegXY = 0;
        if (pNegXNegY < 0) pNegXNegY = 0;

        // Your exact denominators
        double d11 = pX + pY;
        double d10 = pX + pNegY;
        double d01 = pNegX + pY;
        double d00 = pNegX + pNegY;

        return safeTerm(pXY, d11)
                + safeTerm(pXNegY, d10)
                + safeTerm(pNegXY, d01)
                + safeTerm(pNegXNegY, d00);
    }

    @Override
    public String getHeuristicName() {
        return "mutual_information";
    }
}
