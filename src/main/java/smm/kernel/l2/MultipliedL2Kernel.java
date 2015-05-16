package smm.kernel.l2;

import smm.KernelMem;
import smm.data.Document;

public class MultipliedL2Kernel implements L2Kernel {

    private final L2Kernel k1;
    private final L2Kernel k2;

    public MultipliedL2Kernel(L2Kernel k1, L2Kernel k2) {
        this.k1 = k1;
        this.k2 = k2;
    }

    @Override
    public double calc(Document d1, Document d2, KernelMem kss) {
        return k1.calc(d1, d2, kss) * k2.calc(d1, d2, kss);
    }
}
