package smm.kernel.l2;

import java.util.Map.Entry;
import smm.KernelMem;
import smm.data.Document;

public class LinearL2Kernel implements L2Kernel {
    @Override
    public double calc(Document d1, Document d2, KernelMem kss) {
        double ans = 0;
        
        for (Entry<Integer, Integer> e1: d1.freqs.entrySet()) {
            int w1 = e1.getKey();
            int f1 = e1.getValue();
            if (w1 >= kss.wordNum) {
                // out-of-vocabulary
                continue;
            }
            for (Entry<Integer, Integer> e2: d2.freqs.entrySet()) {
                int w2 = e2.getKey();
                int f2 = e2.getValue();
                if (w2 >= kss.wordNum) {
                    // out-of-vocabulary
                    continue;
                }
                ans += f1 * f2 * kss.get(w1, w2);
            }
        }
        
        if (d1.wordCount == 0 || d2.wordCount == 0) {
            ans = 0;
        } else {
            ans /= d1.wordCount * d2.wordCount;
        }

        return ans;

    }
}
