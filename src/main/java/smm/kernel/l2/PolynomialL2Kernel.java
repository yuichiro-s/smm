package smm.kernel.l2;

import smm.KernelMem;
import smm.data.Document;

public class PolynomialL2Kernel implements L2Kernel {
    public final int order;
    public final double c;
    private final LinearL2Kernel linearL2Kernel;

    public PolynomialL2Kernel(int order, double c) {
        this.order = order;
        this.c = c;
        linearL2Kernel = new LinearL2Kernel();
    }

    @Override
    public double calc(Document d1, Document d2, KernelMem kss) {
        return Math.pow(linearL2Kernel.calc(d1, d2, kss) + c, order);
    }
}
