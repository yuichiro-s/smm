/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package smm;

import smm.data.Document;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Io {
    public static List<Document> loadDocuments(Path path) throws IOException {
        List<Document> docs = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, Charset.defaultCharset())) {
            String line;
            while ((line = reader.readLine()) != null) {
                docs.add(new Document(line));
            }
        }
        return docs;
    }

    public static List<double[]> loadWords(Path path) throws IOException {
        List<double[]> ws = new ArrayList<>();
        int dim = -1;
        try (BufferedReader reader = Files.newBufferedReader(path, Charset.defaultCharset())) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.length() == 0) {
                    // empty line
                    ws.add(null);
                } else {
                    String[] es = line.split(" ");
                    // dimentions must be the same
                    assert dim == -1 || es.length == dim;
                    dim = es.length;

                    double[] vec = new double[dim];
                    for (int i = 0; i < dim; i++) {
                        vec[i] = Double.valueOf(es[i]);
                    }
                    ws.add(vec);
                }
            }
        }
        
        return ws;
    }

    public static void writeMatrix(double[][] g, Path path) throws FileNotFoundException, IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {
            for (double[] g1 : g) {
                for (int j = 0; j < g1.length; j++) {
                    writer.write(String.format("%.6e%s", g1[j], j == g1.length-1 ? "\n" : " "));
                }
            }
        }
    }
}
