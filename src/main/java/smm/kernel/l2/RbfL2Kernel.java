package smm.kernel.l2;

import static java.lang.Math.exp;
import java.util.HashMap;
import java.util.Map;
import smm.KernelMem;
import smm.data.Document;

public class RbfL2Kernel implements L2Kernel {

    private final Map<Document, Double> normMem;
    private final LinearL2Kernel linearL2Kernel;

    public final double lambda;

    public RbfL2Kernel(double lambda) {
        normMem = new HashMap<>();
        linearL2Kernel = new LinearL2Kernel();
        this.lambda = lambda;
    }

    @Override
    public double calc(Document d1, Document d2, KernelMem kss) {
        double n1;
        if (normMem.containsKey(d1)) {
            n1 = normMem.get(d1);
        } else {
            n1 = linearL2Kernel.calc(d1, d1, kss);
            normMem.put(d1, n1);
        }

        double n2;
        if (normMem.containsKey(d2)) {
            n2 = normMem.get(d2);
        } else {
            n2 = linearL2Kernel.calc(d2, d2, kss);
            normMem.put(d2, n2);
        }

        double sum = -2 * linearL2Kernel.calc(d1, d2, kss);
        sum += n1;
        sum += n2;
        sum *= -lambda / 2;

        return exp(sum);
    }
    
}
