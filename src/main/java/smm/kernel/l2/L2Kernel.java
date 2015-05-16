package smm.kernel.l2;

import smm.KernelMem;
import smm.data.Document;

public interface L2Kernel {
    public double calc(Document d1, Document d2, KernelMem kss);
}
