package net.openhealthkg.graphminer.heuristics;

/**
 * Implements Mutual Information per Section 13.5.1 eq 13.16 of Manning et al's Introduction to Information Retrieval
 * (2009)
 */
public class MutualInformation extends PXYHeuristic {
    public MutualInformation(long collSize) {
        super(collSize);
    }

    @Override
    public double score() {
        float pXY = getPXandY();
        float pXandNegY = getPXandNegY();
        float pNegXandY = getPNegXandY();
        float pNegXandNegY = getPNegXandNegY();
        return pXY * (Math.log(pXY/(getPX() + getPY())) / Math.log(2)) + // x_1, y_1
            pXandNegY * (Math.log(pXandNegY/(getPX() + getPNegY())) / Math.log(2)) + // x_1, y_0
            pNegXandY * (Math.log(pNegXandY/(getPNegX() + getPY())) / Math.log(2)) + // x_0, y_1
            pNegXandNegY * (Math.log(pNegXandNegY/(getPNegX() + getPNegY())) / Math.log(2)); // x_0, y_0

    }
}
