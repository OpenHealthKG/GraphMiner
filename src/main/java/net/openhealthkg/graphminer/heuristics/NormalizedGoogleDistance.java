package net.openhealthkg.graphminer.heuristics;

/**
 * Implements the Normalized Google Distance (NGD) heuristic, defined as:
 * NGD(x,y) = (max(log f(x), log f(y)) - log f(x,y)) / (log N - min(log f(x), log f(y)))
 * Where:
 *   - f(x)   = number of documents containing x
 *   - f(x,y) = number of documents containing both x and y
 *   - N      = corpus size (total documents)
 */
public class NormalizedGoogleDistance implements PXYHeuristic {

    @Override
    public String getHeuristicName() {
        return "ngd";
    }

    @Override
    public Double call(Long fx, Long fy, Long fxy, Long coll_size) throws Exception {
        // Guardrails
        if (coll_size <= 1.0) return Double.MAX_VALUE;           // log(N) invalid / degenerate corpus
        if (fx <= 0.0 || fy <= 0.0) return Double.MAX_VALUE; // one term never appears (should never happen with our use case)
        if (fxy <= 0.0) return Double.MAX_VALUE;         // never co-occurs

        // Clamp fxy to <= min(fx, fy) to avoid any rounding artifacts violating set logic
        final double fxyClamped = Math.min(fxy, Math.min(fx, fy));

        final double logFx  = Math.log(fx);
        final double logFy  = Math.log(fy);
        final double logFxy = Math.log(fxyClamped);
        final double logN   = Math.log(coll_size);

        final double maxLog = Math.max(logFx, logFy);
        final double minLog = Math.min(logFx, logFy);

        final double denom = (logN - minLog);
        if (denom <= 0.0) return Double.MAX_VALUE; // If term occurs in all documents (unlikely)

        final double ngd = (maxLog - logFxy) / denom;
        return 1/(1 + Math.max(0.0, ngd)); // Re-scale so higher is better
    }
}
