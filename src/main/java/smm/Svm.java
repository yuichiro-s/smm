package smm;

import java.util.List;
import java.util.Map.Entry;
import java.util.logging.Logger;
import smm.data.Document;

public class Svm implements KernelMachine {

    public final Kernel kernel;
    
    public Svm(Kernel kernel) {
        this.kernel = kernel;
    }

    // compute gram matrix
    @Override
    public double[][] gram(List<Document> docs) {
        int docNum = docs.size();

        double gss[][] = new double[docNum][docNum];

        for (int i = 0; i < docNum; i++) {
            Document di = docs.get(i);
            for (int j = i; j < docNum; j++) {
                Document dj = docs.get(j);
                gss[i][j] = gss[j][i] = kernel.calc(di, dj);
            }
        }

        return gss;
    }

    @Override
    public double[][] embeddings(List<Document> trainingDocs, List<Document> testDocs) {
        int docNum = trainingDocs.size();
        int testNum = testDocs.size();

        double es[][] = new double[testNum][docNum];

        for (int i = 0; i < testNum; i++) {
            if (i % 100 == 0) {
                Logger.getGlobal().info(String.format("%d/%d done.", i, testNum));
            }
            Document di = testDocs.get(i);
            for (int j = 0; j < docNum; j++) {
                Document dj = trainingDocs.get(j);
                es[i][j] = kernel.calc(di, dj);
            }
        }

        return es;

    }

    public static interface Kernel {
        double calc(Document d1, Document d2);
    }

    public static class LinearKernel implements Kernel {
        @Override
        public double calc(Document d1, Document d2) {
            // currently O(NM) time
            // can be improved to O(N+M) time by sorting document indices beforehand
            double ans = 0;
            for (Entry<Integer, Integer> e1: d1.freqs.entrySet()) {
                int w1 = e1.getKey();
                int f1 = e1.getValue();
                for (Entry<Integer, Integer> e2: d2.freqs.entrySet()) {
                    int w2 = e2.getKey();
                    int f2 = e2.getValue();
                    if (w1 == w2) {
                        ans += f1 * f2;
                    }
                }
            }

            return ans;
        }
    }

    public static class RbfKernel implements Kernel {
        public final double gamma;

        public RbfKernel(double gamma) {
            this.gamma = gamma;
        }

        @Override
        public double calc(Document d1, Document d2) {
            double ans = 0;
            for (Entry<Integer, Integer> e1: d1.freqs.entrySet()) {
                int w1 = e1.getKey();
                int f1 = e1.getValue();

                if (d2.freqs.containsKey(w1)) {
                    int f2 = d2.freqs.get(w1);
                    ans += (f1-f2) * (f1-f2);
                } else {
                    ans += f1*f1;
                }
            }

            for (Entry<Integer, Integer> e2: d2.freqs.entrySet()) {
                int w2 = e2.getKey();
                int f2 = e2.getValue();
                if (!d1.freqs.containsKey(w2)) {
                    ans += f2*f2;
                }
            }
            
            ans *= -0.5 * gamma;
            return Math.exp(ans);
        }
    }
    
}
