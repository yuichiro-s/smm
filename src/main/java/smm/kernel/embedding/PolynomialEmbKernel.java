package smm.kernel.embedding;

public class PolynomialEmbKernel implements EmbeddingKernel {
    public final int order;
    public final double c;

    public PolynomialEmbKernel(int order, double c) {
        this.order = order;
        this.c = c;
    }

    @Override
    public double calc(double[] w1, double[] w2) {
        if (w1 != null && w2 != null) {
            double sum = 0;
            for (int i = 0; i < w1.length; i++) {
                sum += w1[i] * w2[i];
            }
            return Math.pow(sum+c, order);
        } else {
            return 0;
        }
    }
}
