package smm.kernel.embedding;

public class RbfEmbKernel implements EmbeddingKernel {

    public final double gamma;

    public RbfEmbKernel(double gamma) {
        this.gamma = gamma;
    }

    @Override
    public double calc(double[] w1, double[] w2) {
        if (w1 != null && w2 != null) {
            double sqsum = 0;
            for (int i = 0; i < w1.length; i++) {
                double diff = w1[i] - w2[i];
                sqsum += diff * diff;
            }
            sqsum *= -0.5 * gamma;

            return Math.exp(sqsum);
        } else {
            return 0;
        }
    }
    
}
