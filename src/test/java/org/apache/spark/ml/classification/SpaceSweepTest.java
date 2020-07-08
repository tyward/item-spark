package org.apache.spark.ml.classification;

import edu.columbia.tjw.item.spark.ClassificationModelEvaluator;
import edu.columbia.tjw.item.util.random.PrngType;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.TestUtil.GeneratedData;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Random;

class SpaceSweepTest
{
    private static final int STATUS_COUNT = 3;

    private static final int MAX_ITER = 500;
    private static final int BLOCK_SIZE = 4 * 1024;

    private static final String SOLVER = "l-bfgs";
    private static final String SOLVER_GD = "gd";
    private static final long PRNG_SEED = 0xabcdef123456L;
    private static final double TOLERANCE = 1.0e-8;
    private static final double STEP_SIZE = 0.03;


    //@Test
    void testSweep() throws Exception
    {
        final SparkSession spark = TestUtil.generateSparkSession();
        final Dataset<Row> frame = TestUtil.generateData(spark);

        final Random rand = RandomTool.getRandom(PrngType.SECURE, PRNG_SEED);

        final int[][] testLayers = new int[][]{
                {TestUtil.INPUT_COLS.size(), STATUS_COUNT},
                {TestUtil.INPUT_COLS.size(), 5, STATUS_COUNT},
                {TestUtil.INPUT_COLS.size(), 8, 5, STATUS_COUNT},
                {TestUtil.INPUT_COLS.size(), TestUtil.INPUT_COLS.size(), 8, 5, STATUS_COUNT},
        };

        final int[] sampleSizes = new int[]{
                128, 256, 512, 1024, 2048,
                4096, 8 * 1024, 16 * 1024, 32 * 1024,
                64 * 1024,
                128 * 1024};

        final int repCount = 10;

        try (final OutputStream oStream = new FileOutputStream("/Users/tyler/Desktop/runResults.csv");
             PrintStream output = new PrintStream(oStream))
        {
            TestUtil.printHeader(output);

            for (int k = 0; k < testLayers.length; k++)
            {
                for (int w = 0; w < sampleSizes.length; w++)
                {
                    for (int i = 0; i < repCount; i++)
                    {
                        final long prngSeed = rand.nextLong();
                        final GeneratedData data = TestUtil.prepareData(frame, prngSeed, sampleSizes[w]);

                        final ClassificationModelEvaluator.EvaluationResult startingPoint = generateMleResult(data,
                                prngSeed, testLayers[k], null, SOLVER);
                        TestUtil.printResults(startingPoint, output);

                        // ICE after here.
                        {
                            final ClassificationModelEvaluator.EvaluationResult iceResult = generateIceResult(data,
                                    prngSeed, testLayers[k], null, SOLVER);
                            TestUtil.printResults(iceResult, output);
                        }
                    }
                }
            }
        }
    }


    private ClassificationModelEvaluator.EvaluationResult generateMleResult(final GeneratedData data_,
                                                                            final long prngSeed_,
                                                                            final int[] layers_,
                                                                            final Vector startingPoint_,
                                                                            final String solver_)
    {
        MultilayerPerceptronClassifier mlpFitter = new MultilayerPerceptronClassifier().setLabelCol("NEXT_STATUS");
        mlpFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(BLOCK_SIZE).setSolver(solver_).setTol(TOLERANCE)
                .setStepSize(STEP_SIZE);

        if (null != startingPoint_)
        {
            mlpFitter.setInitialWeights(startingPoint_);
        }

        return ClassificationModelEvaluator
                .evaluate(mlpFitter, "MLE[" + (startingPoint_ == null) + "][" + solver_ + "]",
                        data_.getFitting(), data_.getTesting(),
                        prngSeed_,
                        layers_);
    }

    private ClassificationModelEvaluator.EvaluationResult generateIceResult(GeneratedData data_,
                                                                            final long prngSeed_,
                                                                            final int[] layers_,
                                                                            final Vector startingPoint_,
                                                                            final String solver_)
    {
        IcePerceptronClassifier iceFitter = new IcePerceptronClassifier().setLabelCol("NEXT_STATUS");
        iceFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(BLOCK_SIZE).setSolver(solver_).setTol(TOLERANCE)
                .setStepSize(STEP_SIZE);

        if (null != startingPoint_)
        {
            iceFitter.setInitialWeights(startingPoint_);
        }

        return ClassificationModelEvaluator
                .evaluate(iceFitter,
                        "ICE[" + (startingPoint_ == null) + "][" + solver_ + "]", data_.getFitting(),
                        data_.getTesting(),
                        prngSeed_,
                        layers_);
    }

//    private void validateGradients(IcePerceptronClassifier iceFitter, final GeneratedData data,
//                                   IcePerceptronClassificationModel model)
//    {
//        final double[] weights = model.weights().toArray().clone();
//
//        final int size = weights.length;
//
//        final double[] grad = new double[size];
//        final double[] jDiag = new double[size];
//
//        final double[] fdGrad = new double[size];
//        final double[] fdDiag = new double[size];
//
//        final Dataset<Row> fittingData = data.getFitting().limit(1);
//
//
//        final double loss = model.computeGradients(fittingData, model.weights(), grad, jDiag);
//
//        for (int i = 0; i < weights.length; i++)
//        {
//            final double[] w2 = weights.clone();
//
//            final double h = Math.max(Math.abs(w2[i] * 1.0e-4), 1.0e-12);
//            w2[i] += h;
//
//            final double[] g2 = new double[size];
//            final double[] jDiag2 = new double[size];
//
//            final double lossUp = model.computeGradients(fittingData, Vectors.dense(w2), g2, jDiag2);
//
//            final double fdd = (lossUp - loss) / h;
//
//            final double origGrad = grad[i];
//            final double shiftGrad = g2[i];
//            final double shiftDiag = jDiag2[i];
//            final double origDiag = jDiag[i];
//
//            final double fdd2 = (shiftGrad - origGrad) / h;
//
//            fdGrad[i] = fdd;
//            fdDiag[i] = fdd2;
//
//            w2[i] = weights[i] - h;
//            final double lossDown = model.computeGradients(fittingData, Vectors.dense(w2), g2, jDiag2);
//
//            final double fdd2a = (lossUp + lossDown - (2 * loss)) / (h * h);
//            final double fdd2b = (origGrad - g2[i]) / h;
//
//
//            System.out.println("FDD[" + origGrad + "]: " + fdd);
//            System.out.println("FDD2[" + origDiag + ", " + shiftDiag + "]: " + fdd2);
//            System.out.println("next.");
//        }
//
//        final double fdCos = MathTools.cos(fdGrad, grad);
//        final double fdCos2 = MathTools.cos(fdDiag, jDiag);
//
//        final double magRatio = MathTools.magnitude(fdGrad) / MathTools.magnitude(grad);
//        final double magRatio2 = MathTools.magnitude(fdDiag) / MathTools.magnitude(jDiag);
//
//        final double[] fdGradA = new double[INPUT_COLS.length];
//        final double[] fdDiagA = new double[INPUT_COLS.length];
//
//
//        for (int i = 0; i < INPUT_COLS.length; i++)
//        {
//            // These are normalize, so the same H should work fine for everything.
//            final double h = 1.0e-6;
////            final double scaleUp = 1.0 + h;
////            final double scaleDown = 1.0 - h;
//
//            // Now update the contents of the row....
//            final Dataset<Row> fdUp = assemble(fittingData.withColumn(INPUT_COLS[i],
//                    functions.expr(INPUT_COLS[i] + " + " + h)));
//            final Dataset<Row> fdDown = assemble(fittingData.withColumn(INPUT_COLS[i],
//                    functions.expr(INPUT_COLS[i] + " - " + h)));
//
//            final double[] g2 = new double[size];
//            final double[] jDiag2 = new double[size];
//
//            final double lossUp = model.computeGradients(fdUp, Vectors.dense(weights), g2, jDiag2);
//            final double lossDown = model.computeGradients(fdDown, Vectors.dense(weights), g2, jDiag2);
//
//            final double fdd = (lossUp - loss) / h;
//            fdGradA[i] = fdd;
//
//            final double fddDown = (loss - lossDown) / h;
//
//            final double fdd2 = (lossUp + lossDown - (2 * loss)) / (h * h);
//            fdDiagA[i] = fdd2;
//        }
//
//        final double[] grad2 = new double[size];
//        final double[] jDiag2 = new double[size];
//        final double loss2 = model.computeGradients(fittingData, model.weights(), grad2, jDiag2);
//
//        System.out.println("Cos[" + magRatio + "]: " + fdCos);
//        System.out.println("Cos2[" + magRatio2 + "]: " + fdCos2);
//        System.out.println("Done.");
//
//    }
//
//
}