package smm.kernel.l2;

import java.util.Map.Entry;
import smm.KernelMem;
import smm.data.Document;

// linear kernel between two documents
public class BowL2Kernel implements L2Kernel {
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
            if (d2.freqs.containsKey(w1)) {
                ans += f1 * d2.freqs.get(w1);
            }
        }

        return ans;
    }
}
