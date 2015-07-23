/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package smm.data;

import java.util.HashMap;

public class Document {
    public final HashMap<Integer, Integer> freqs; // word frequencies
    public final int wordCount;   // # of freqs in the document

    public Document(String line) {
        freqs = new HashMap<>();
        int wc = 0;

        String[] es = line.split(" ");
        for (int i = 1; i < es.length; i++) {
            String[] i_f = es[i].split(":");
            int wordIndex = Integer.valueOf(i_f[0])-1;
            int wordFreq = Integer.valueOf(i_f[1]);
            freqs.put(wordIndex, wordFreq);
            wc += wordFreq;
        }

        wordCount = wc;
    }

    public Document(HashMap<Integer, Integer> freqs, int wordCount) {
        this.freqs = new HashMap<>(freqs);
        this.wordCount = wordCount;
    }
    
}
