package net.openhealthkg.graphminer.heuristics;

/**
 * Implements the Normalized Google Distance (NGD) heuristic, defined as:
 * NGD(x,y) = (max(log f(x), log f(y)) - log f(x,y)) / (log N - min(log f(x), log f(y)))
 * Where:
 *   - f(x)   = number of documents containing x
 *   - f(x,y) = number of documents containing both x and y
 *   - N      = corpus size (total documents)*
 * A lower score indicates a greater degree of relatedness between x and y
 */
public class NormalizedGoogleDistance extends PXYHeuristic {

    public NormalizedGoogleDistance(long collSize) {
        super(collSize);
    }

    @Override
    public double score() {
        final double N = getCollSize();

        final double fx  = this.x_1.size();
        final double fy  = this.y_1.size();
        final double fxy = Math.max(0.0, Math.rint(getPXandY() * N));

        // Guardrails
        if (N <= 1.0) return Double.MAX_VALUE;           // log(N) invalid / degenerate corpus
        if (fx <= 0.0 || fy <= 0.0) return Double.MAX_VALUE; // one term never appears (should never happen with our use case)
        if (fxy <= 0.0) return Double.MAX_VALUE;         // never co-occurs

        // Clamp fxy to <= min(fx, fy) to avoid any rounding artifacts violating set logic
        final double fxyClamped = Math.min(fxy, Math.min(fx, fy));

        final double logFx  = Math.log(fx);
        final double logFy  = Math.log(fy);
        final double logFxy = Math.log(fxyClamped);
        final double logN   = Math.log(N);

        final double maxLog = Math.max(logFx, logFy);
        final double minLog = Math.min(logFx, logFy);

        final double denom = (logN - minLog);
        if (denom <= 0.0) return Double.MAX_VALUE; // If term occurs in all documents (unlikely)

        final double ngd = (maxLog - logFxy) / denom;
        return Math.max(0.0, ngd);
    }
}
