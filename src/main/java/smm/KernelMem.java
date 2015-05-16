package smm;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import smm.kernel.embedding.EmbeddingKernel;

public class KernelMem {

    public final EmbeddingKernel kernel;
    public final List<double[]> wordVecs;

    private final double[][] mem;
    private final boolean[][] stored;

    public final int wordNum;

    public KernelMem(EmbeddingKernel kernel, List<double[]> wordVecs) {
        this.kernel = kernel;
        this.wordVecs = wordVecs;

        this.wordNum = wordVecs.size();

        mem = new double[wordNum][wordNum];
        stored = new boolean[wordNum][wordNum];
    }

    public double get(int i, int j) {
        if (stored[i][j]) {
            return mem[i][j];
        } else {
            double k = this.kernel.calc(wordVecs.get(i), wordVecs.get(j));
            mem[i][j] = mem[j][i] = k;
            stored[i][j] = stored[j][i] = true;

            return k;
        }
    }

    public void printUsage() {
        long count = 0;
        long total = 0;
        for (int i = 0; i < wordNum; i++) {
            for (int j = 0; j < wordNum; j++) {
                if (stored[i][j]) count++;
                total++;
            }
        }
        System.err.println(count*100/total + "% used.");
    }

    public void printKernel(Path path) throws IOException {
        // force computation for all pairs
        for (int i = 0; i < wordNum; i++) {
            for (int j = i; j < wordNum; j++) {
                if (!stored[i][j]) {
                    get(i, j);
                }
            }
        }

        Io.writeMatrix(mem, path);
    }
    
}