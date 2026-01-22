package net.openhealthkg.graphminer.heuristics;

import java.io.Serializable;
import java.util.HashSet;

/**
 * Denotes a heuristic that uses variations of P(X_1), P(X_0), P(Y_1), P(Y_0), and P(X_1^Y_1) to mine potential graph
 * edges between nodes X and Y within an overall corpus
 */
public abstract class PXYHeuristic implements Serializable {
    private HashSet<String> x_1 = new HashSet<>();
    private HashSet<String> y_1 = new HashSet<>();
    private long collSize;

    public PXYHeuristic(long collSize) {
        this.collSize = collSize;
    }

    public PXYHeuristic merge(PXYHeuristic other) {
        x_1.addAll(other.x_1);
        y_1.addAll(other.y_1);
        return this;
    }

    public float getPX() {
        return (float) x_1.size() / collSize;
    }

    public float getPNegX() {
        return (float) (collSize - x_1.size()) / collSize;
    }

    public float getPY() {
        return (float) y_1.size() / collSize;
    }

    public float getPNegY() {
        return (float) (collSize - y_1.size()) / collSize;
    }

    public float getPXandY() {
        HashSet<String> x_1y_1 = new HashSet<>(x_1);
        x_1y_1.retainAll(y_1);
        return (float) x_1y_1.size() / collSize;
    }

    public float getPNegXandY() {
        HashSet<String> x_0y_1 = new HashSet<>(y_1);
        x_0y_1.removeAll(x_1);
        return (float) x_0y_1.size() / collSize;
    }

    public float getPXandNegY() {
        HashSet<String> x_1y_0 = new HashSet<>(x_1);
        x_1y_0.removeAll(y_1);
        return (float) x_1y_0.size() / collSize;
    }

    public float getPNegXandNegY() {
        HashSet<String> x_1y_1 = new HashSet<>(x_1);
        x_1y_1.retainAll(y_1);
        return (float) (collSize - x_1y_1.size()) / collSize;
    }

    public float getCollSize() {
        return collSize;
    }

    public abstract double score();



}
