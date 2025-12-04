package app;

import java.util.Scanner;
import ml.core.*;
import ml.metrics.*;
import ml.models.*;
import ml.preprocess.*;
import ml.helpers.*;
import java.io.IOException;

public class Main {
    static Dataset TRAIN, TEST;

    public static void main(String[] args) {
        ArgParser ap = new ArgParser(args);

        try (Scanner in = new Scanner(System.in)) {
            while (in.hasNextLine()) {
                int opt = 0;

                try {
                    String line = in.nextLine().trim();
                    if (line.isEmpty()) continue;
                    opt = Integer.parseInt(line);
                } catch (NumberFormatException e) {
                    continue;
                }
                System.out.println("DEBUG: processing option " + opt);
                System.out.flush();

                if(opt==8) return;

                if(opt==1) {
                    try { loadData(ap); }
                    catch (Exception e) { System.out.println("Error loading data: " + e.getMessage()); }
                }
                if(opt==2) runLinear(ap);
                if(opt==3) runLogistic(ap);
                if(opt==4) runKNN(ap);
                if(opt==5) runTree(ap);
                if(opt==6) runGNB(ap);
                if(opt==7) printResults();

                System.out.flush()
            }

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // --- Helper to ensure data is loaded before running algorithms ---
    private static void ensureDataLoaded(ArgParser ap) throws Exception {
        if (TRAIN == null) {
            loadData(ap);
        }
    }

    private static void loadData(ArgParser ap) throws Exception {
        String path = ap.get("train", "../data/adult_income_cleaned.csv");
        boolean normalize = ap.has("normalize");
        System.out.println("Loading "+path+" (normalize="+normalize+") ...");
        long t0 = System.nanoTime();
        // Uses the classification loader by default for options 3,4,5,6
        Dataset.TrainTest tt = AdultPipeline.loadClassification(path, "income", normalize, 0.2, 42);
        TRAIN = tt.train; TEST = tt.test;
        double secs = (System.nanoTime()-t0)/1e9;
        System.out.printf("Loaded: %d train / %d test / %d features in %.3fs%n",
                TRAIN.nRows(), TEST.nRows(), TRAIN.nCols(), secs);
    }

    private static void runLinear(ArgParser args) throws IOException {
        String trainPath = args.get("train", "../data/adult_income_cleaned.csv");
        String target = args.get("target", "hours.per.week");
        boolean normalize = args.has("normalize");
        double l2 = args.getDouble("l2", 1.0);
        long seed = (long) args.getDouble("seed", 42.0);

        System.out.println("\nLinear Regression (closed-form):");
        System.out.println("********************************");
        System.out.println("Input option 1: Target variable: " + target);
        System.out.println("Input option 2: L2 = " + l2);

        Timer t = new Timer();
        t.start();
        Dataset.TrainTest tt = AdultPipeline.loadRegression(trainPath, target, normalize, 0.2, seed);
        double loadSecs = t.seconds();
        System.out.printf("Loaded regression dataset in %.3f s%n", loadSecs);

        LinearRegression model = new LinearRegression(l2);
        t.start();
        model.fit(tt.train);
        double trainSecs = t.seconds();

        double[] yhat = model.predict(tt.test.X);
        double rmse = Metrics.rmse(tt.test.y, yhat);
        double r2   = Metrics.r2(tt.test.y, yhat);

        int sloc = SLOC.forClass(LinearRegression.class, ".");
        System.out.printf("Train time: %.4f s%n", trainSecs);
        System.out.printf("RMSE: %.4f%n", rmse);
        System.out.printf("R^2: %.4f%n", r2);
        System.out.println("SLOC: " + sloc + "\n");

        log(model.name(), trainSecs, "RMSE", rmse, "R^2",  r2, sloc);
    }

    private static void runLogistic(ArgParser ap){
        System.out.println("DEBUG");
        System.out.flush();

        // Ensure data is loaded
        try { ensureDataLoaded(ap); } catch (Exception e) { System.out.println(e.getMessage()); return; }

        System.out.println("DEBUG: Data");

        double lr = ap.getDouble("lr", 0.001);
        int epochs = ap.getInt("epochs", 3000);
        double l2 = ap.getDouble("l2", 1.0);
        long seed = 7;
        String srcRoot = ap.get("srcroot", ".");

        System.out.printf("DEBUG: lr=%.4f, epochs=%d, l2=%.4f%n", lr, epochs, l2);
        System.out.flush();

        try {

            Timer t = new Timer();
            t.start();
            LogisticRegression m = new LogisticRegression(lr, epochs, l2, seed);

            System.out.println("DEBUG: Staring fit");
            m.fit(TRAIN);
            double secs = t.seconds();
            System.out.println("DEBUG: fit complete");

            double[] pred = m.predict(TEST.X);
            double acc = Metrics.accuracy(TEST.y, pred);
            double f1  = Metrics.macroF1(TEST.y, pred);
            int sloc = SLOC.forClass(LogisticRegression.class, srcRoot);

            System.out.println("\nLogistic Regression (closed-form):");
            System.out.println("********************************");
            System.out.println("Input option 1: Lr = " + lr);
            System.out.println("Input option 2: epoch = " + epochs);
            System.out.println("Input option 3: L2 = " + l2);
            System.out.printf("Train time: %.4f s%n", secs);
            System.out.printf("Accuracy: %.4f%n", acc);
            System.out.printf("Macro-F1: %.4f%n", f1);
            System.out.println("SLOC: " + sloc + "\n");
            System.out.flush();
            log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
        } catch (Exception e) {
            System.out.println("Error in run Logsitic: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runKNN(ArgParser ap){
        // Ensure data is loaded
        try { ensureDataLoaded(ap); } catch (Exception e) { System.out.println(e.getMessage()); return; }

        int k = ap.getInt("k", 15);
        String srcRoot = ap.get("srcroot", ".");

        String distance = ap.get("distance", "euclidean");
        boolean weighted = ap.has("weighted");
        String tie = ap.get("tie", "random");

        Timer t = new Timer(); t.start();
        KNN m = new KNN(k, distance, weighted, tie, 0L);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);

        int sloc = SLOC.forClass(KNN.class, srcRoot);

        System.out.println("\nK-Nearest Neightbors:");
        System.out.println("********************************");
        System.out.println("K = " + k);
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", acc);
        System.out.printf("Macro-F1: %.4f%n", f1);
        System.out.println("SLOC: " + sloc + "\n");
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }

    private static void runTree(ArgParser ap){
        // Ensure data is loaded
        try { ensureDataLoaded(ap); } catch (Exception e) { System.out.println(e.getMessage()); return; }

        int maxDepth   = ap.getInt("max_depth", 5);
        int minSamples = ap.getInt("min_samples", 10);
        int nBins      = ap.getInt("bins", 16);
        String srcRoot = ap.get("srcroot", ".");

        Timer t = new Timer(); t.start();
        DecisionTree m = new DecisionTree(maxDepth, 10, nBins);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);
        int sloc = SLOC.forClass(DecisionTree.class, srcRoot);

        System.out.println("\nDecision Tree:");
        System.out.println("********************************");
        System.out.println("max_depth = " + maxDepth);
        System.out.println("min_samples = " + minSamples);
        System.out.println("bins = " + nBins);
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", acc);
        System.out.printf("Macro-F1: %.4f%n", f1);
        System.out.println("SLOC: " + sloc + "\n");
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }

    private static void runGNB(ArgParser ap){
        // Ensure data is loaded
        try { ensureDataLoaded(ap); } catch (Exception e) { System.out.println(e.getMessage()); return; }

        double smooth = ap.getDouble("smoothing", 1e-1);
        String srcRoot = ap.get("srcroot", ".");

        Timer t = new Timer(); t.start();
        GaussianNaiveBayes m = new GaussianNaiveBayes(smooth);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);
        int sloc = SLOC.forClass(GaussianNaiveBayes.class, srcRoot);

        System.out.println("\nGaussian Naive Bayes:");
        System.out.println("********************************");
        System.out.println("Smoothing = " + smooth);
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", acc);
        System.out.printf("Macro-F1: %.4f%n", f1);
        System.out.println("SLOC: " + sloc + "\n");
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }

    static class Row {
        String impl, algo; double t, m1, m2; String m1n, m2n; int sloc;
        Row(String impl, String algo, double t, String m1n, double m1, String m2n, double m2, int sloc){
            this.impl=impl; this.algo=algo; this.t=t; this.m1n=m1n; this.m1=m1; this.m2n=m2n; this.m2=m2; this.sloc=sloc;
        }
    }
    static java.util.List<Row> RESULTS = new java.util.ArrayList<>();
    static void log(String algo, double secs, String m1n, double m1, String m2n, double m2, int sloc){
        RESULTS.add(new Row("Java", algo, secs, m1n, m1, m2n, m2, sloc));
    }
    static void printResults() {
        System.out.println("General Results (Comparison):");
        System.out.printf("%-8s %-26s %10s  %-10s %10s  %-10s %10s  %6s%n",
                "Impl","Algorithm","TrainTime","Metric1","Value1","Metric2","Value2","SLOC");
        System.out.println("-------------------------------------------------------------------------------------------------------------------");
        for (Row r : RESULTS) {
            System.out.printf("%-8s %-26s %8.4fs  %-10s %10.4f  %-10s %10.4f  %6d%n",
                    r.impl, r.algo, r.t, r.m1n, r.m1, r.m2n, r.m2, r.sloc);
        }
    }
}
