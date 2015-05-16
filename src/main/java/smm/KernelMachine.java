package smm;

import java.util.List;
import smm.data.Document;

public interface KernelMachine {
    double[][] gram(List<Document> docs);
    double[][] embeddings(List<Document> trainingDocs, List<Document> testDocs);
}
