package net.openhealthkg.graphminer.heuristics;

public class ChiSquared implements PXYHeuristic {

    @Override
    public String getHeuristicName() {
        return "chi_squared";
    }

    @Override
    public Double call(Long fx, Long fy, Long fxy, Long coll_size) throws Exception {
        double N = coll_size;

        // Get counts from probabilities to construct the chi-squared table
        double a = fxy;                         // X ^ Y
        double b = fx-fxy;                      // X ^ !Y
        double c = fy-fxy;                      // !X ^ Y
        double d = coll_size - a - b - c;       // !X ^ !Y

        // Row/Column Edges to calculate Expected Counts (Row Total * Column Total/N)
        double rowX = a + b;
        double rowNotX = c + d;
        double colY = a + c;
        double colNotY = b + d;

        // Expected counts
        double ea = (rowX * colY) / N;
        double eb = (rowX * colNotY) / N;
        double ec = (rowNotX * colY) / N;
        double ed = (rowNotX * colNotY) / N;

        double chi2 = 0.0;

        // Use zero-guard to prevent value blowup on sparse data
        if (ea > 0) chi2 += Math.pow(a - ea, 2) / ea;
        if (eb > 0) chi2 += Math.pow(b - eb, 2) / eb;
        if (ec > 0) chi2 += Math.pow(c - ec, 2) / ec;
        if (ed > 0) chi2 += Math.pow(d - ed, 2) / ed;
        return chi2;
    }
}
