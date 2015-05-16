package smm;

import smm.data.Document;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import smm.kernel.embedding.EmbeddingKernel;
import smm.kernel.embedding.LinearEmbKernel;
import smm.kernel.embedding.PolynomialEmbKernel;
import smm.kernel.embedding.RbfEmbKernel;
import smm.kernel.l2.BowL2Kernel;
import smm.kernel.l2.L2Kernel;
import smm.kernel.l2.LinearL2Kernel;
import smm.kernel.l2.MultipliedL2Kernel;
import smm.kernel.l2.RbfL2Kernel;
import smm.kernel.l2.PolynomialL2Kernel;

public class Main {

    @SuppressWarnings("static-access")
    public static void main(String[] args) throws IOException {
        Logger logger = Logger.getGlobal();
        logger.setLevel(Level.ALL);

        Options opts = new Options();

        opts.addOption(OptionBuilder
                .withArgName("kernel")
                .hasArgs()
                .withDescription("use SVM kernel")
                .create("svm"));

        opts.addOption(OptionBuilder
                .withArgName("kernel")
                .hasArgs()
                .withDescription("embedding kernel")
                .create("emb"));
        opts.addOption(OptionBuilder
                .withArgName("kernel")
                .hasArgs()
                .withDescription("level-2 kernel")
                .create("l2"));

        opts.addOption(OptionBuilder
                .withDescription("regard kernel parameters as relative values and multiply them by average norm of word vectors")
                .create("relative"));

        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("word vectors")
                .create("vec"));

        opts.addOption(OptionBuilder
                .withDescription("multiply by linear BoW kernel")
                .create("bow"));

        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("training data")
                .isRequired()
                .create("train"));
        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("test data")
                .create("test"));

        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("output for Gram matrix")
                .isRequired()
                .create("gram"));
        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("output for test data")
                .create("testout"));

        opts.addOption(OptionBuilder
                .withArgName("file")
                .hasArg()
                .withDescription("output for Gram matrix for words")
                .create("kss"));

        HelpFormatter f = new HelpFormatter();
        String usage = "smm [-emb LIN|POLY <N> <C>|RBF <gamma>|URBF -l2 LIN|POLY <N> <C>|RBF <lambda> [-bow] [-relative]] [-svm LIN|RBF <gamma>] -train <file> -gram <file> [-kss <file>] [-test <file> -testout <file>]";

        KernelMachine km = null;

        CommandLineParser parser = new BasicParser();
        try {
            CommandLine cmd = parser.parse(opts, args);

            Path trainPath = Paths.get(cmd.getOptionValue("train"));
            Path gramPath = Paths.get(cmd.getOptionValue("gram"));

            logger.log(Level.INFO, "Loading training data from {0} ...", trainPath.toString());
            List<Document> trainDocs = Io.loadDocuments(trainPath);

            if (cmd.hasOption("svm")) {
                // use SVM
                String[] kernelStr = cmd.getOptionValues("svm");
                Svm.Kernel kernel = null;
                switch (kernelStr[0]) {
                    case "LIN": {
                        kernel = new Svm.LinearKernel();
                        break;
                    }
                    case "RBF": {
                        kernel = new Svm.RbfKernel(Double.valueOf(kernelStr[1]));
                        break;
                    }
                    default: {
                        throw new UnsupportedOperationException(kernelStr[0]);
                    }
                }
                km = new Svm(kernel);
            } else {
                // use SMM
                Path vecPath = Paths.get(cmd.getOptionValue("vec"));
                List<double[]> wordVecs = Io.loadWords(vecPath);

                double avg_sq_norm = 0.0;
                if (cmd.hasOption("relative")) {
                    int n = 0;
                    for (double[] v: wordVecs) {
                        if (v != null) {
                            for (double d: v) {
                                avg_sq_norm += d*d;
                            }
                            n++;
                        }
                    }
                    avg_sq_norm /= n;
                }

                String[] embKernelStr = cmd.getOptionValues("emb");
                EmbeddingKernel embKernel = null;
                switch (embKernelStr[0]) {
                    case "LIN": {
                        embKernel = new LinearEmbKernel();
                        break;
                    }
                    case "RBF": {
                        double gamma = Double.valueOf(embKernelStr[1]);
                        if (cmd.hasOption("relative")) {
                            gamma /= avg_sq_norm;
                        }
                        embKernel = new RbfEmbKernel(gamma);
                        break;
                    }
                    case "POLY": {
                        int order = Integer.valueOf(embKernelStr[1]);
                        double c = Double.valueOf(embKernelStr[2]);
                        if (cmd.hasOption("relative")) {
                            c *= Math.sqrt(avg_sq_norm);
                        }
                        embKernel = new PolynomialEmbKernel(order, c);
                        break;
                    }
                    default: {
                        throw new UnsupportedOperationException(embKernelStr[0]);
                    }
                }

                String[] l2KernelStr = cmd.getOptionValues("l2");
                L2Kernel l2Kernel = null;
                switch (l2KernelStr[0]) {
                    case "LIN": {
                        l2Kernel = new LinearL2Kernel();
                        break;
                    }
                    case "RBF": {
                        l2Kernel = new RbfL2Kernel(Double.valueOf(l2KernelStr[1]));
                        break;
                    }
                    case "POLY": {
                        int order = Integer.valueOf(l2KernelStr[1]);
                        double c = Double.valueOf(l2KernelStr[2]);
                        l2Kernel = new PolynomialL2Kernel(order, c);
                        break;
                    }
                    default: {
                        throw new UnsupportedOperationException(l2KernelStr[0]);
                    }
                }
                
                if (cmd.hasOption("bow")) {
                    l2Kernel = new MultipliedL2Kernel(l2Kernel, new BowL2Kernel());
                }

                Smm smm = new Smm(embKernel, l2Kernel, wordVecs);
                km = smm;

                if (cmd.hasOption("kss")) {
                    smm.kss.printKernel(Paths.get(cmd.getOptionValue("kss")));
                }
            }

            logger.info("Calculating Gram matrix...");
            double[][] g = km.gram(trainDocs);

            logger.log(Level.INFO, "Writing Gram matrix to {0} ...", gramPath.toString());
            Io.writeMatrix(g, gramPath);

            if (cmd.hasOption("test")) {
                Path testPath = Paths.get(cmd.getOptionValue("test"));
                if (!cmd.hasOption("testout")) {
                    throw new RuntimeException("Please specify output for test data.");
                }
                Path testOutPath = Paths.get(cmd.getOptionValue("testout"));

                logger.log(Level.INFO, "Loading test data from {0} ...", testPath.toString());
                List<Document> testDocs = Io.loadDocuments(testPath);

                logger.info("Calculating kernel values for test data...");
                double[][] es = km.embeddings(trainDocs, testDocs);

                logger.log(Level.INFO, "Writing kernel values for test data to {0} ...", testOutPath.toString());
                Io.writeMatrix(es, testOutPath);
            }
            
        } catch (ParseException ex) {
            ex.printStackTrace();
            f.printHelp(usage, opts);
        }

    }
    
}
