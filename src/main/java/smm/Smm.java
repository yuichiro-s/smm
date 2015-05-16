package smm;

import smm.data.Document;
import java.util.List;
import java.util.logging.Logger;
import smm.kernel.embedding.EmbeddingKernel;
import smm.kernel.l2.L2Kernel;

public class Smm implements KernelMachine {
    public final EmbeddingKernel embeddingKernel;
    public final L2Kernel l2Kernel;
    public final List<double[]> wordVecs;

    public final KernelMem kss;

    public Smm(EmbeddingKernel embeddingKernel, L2Kernel l2Kernel, List<double[]> wordVecs) {
        this.embeddingKernel = embeddingKernel;
        this.l2Kernel = l2Kernel;
        this.wordVecs = wordVecs;

        kss = new KernelMem(embeddingKernel, wordVecs);
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

                gss[i][j] = gss[j][i] = l2Kernel.calc(di, dj, kss);
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
                //kss.printUsage();
                Logger.getGlobal().info(String.format("%d/%d done.", i, testNum));
            }
            Document di = testDocs.get(i);
            for (int j = 0; j < docNum; j++) {
                Document dj = trainingDocs.get(j);
                es[i][j] = l2Kernel.calc(di, dj, kss);
            }
        }

        return es;

    }
 
}
