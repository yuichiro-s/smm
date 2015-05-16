package smm.kernel.embedding;

public class LinearEmbKernel implements EmbeddingKernel {
    @Override
    public double calc(double[] w1, double[] w2) {
        if (w1 != null && w2 != null) {
            double sum = 0;
            for (int i = 0; i < w1.length; i++) {
                sum += w1[i] * w2[i];
            }
            return sum;
        } else {
            return 0;
        }
    }
}
