package org.apache.spark.ml.classification;

import edu.columbia.tjw.item.spark.ClassificationModelEvaluator;
import org.apache.spark.ml.classification.TestUtil.GeneratedData;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.PrintStream;

class IcePerceptronClassifierTest
{
    private static final int STATUS_COUNT = 3;


    private static final int[] LAYERS = new int[]{TestUtil.INPUT_COLS.size(), 2, STATUS_COUNT};
    private static final int MAX_ITER = 500;
    private static final int BLOCK_SIZE = 4 * 1024;
    private static final int SAMPLE_SIZE = 4 * BLOCK_SIZE;

    private static final String SOLVER = "l-bfgs";
    private static final String SOLVER_GD = "gd";
    private static final long PRNG_SEED = 0xabcdef123456L;
    private static final double TOLERANCE = 1.0e-8;
    private static final double STEP_SIZE = 0.03;

    @Test
    void testICE()
    {
        final SparkSession spark = TestUtil.generateSparkSession();
        final Dataset<Row> frame = TestUtil.generateData(spark);
        final TestUtil.GeneratedData data = TestUtil.prepareData(frame, PRNG_SEED, SAMPLE_SIZE);
        final ClassificationModelEvaluator.EvaluationResult result = generateIceResult(data, PRNG_SEED,
                LAYERS, null, SOLVER);

        PrintStream output = System.out;
        TestUtil.printHeader(output);
        TestUtil.printResults(result, output);

        Assertions.assertEquals(0.2043469943553112, result.getFittingEntropy().getCrossEntropy());
        Assertions.assertEquals(0.19998376076675636, result.getTestingEntropy().getCrossEntropy());
    }

    @Test
    void testMLP()
    {
        final SparkSession spark = TestUtil.generateSparkSession();
        final Dataset<Row> frame = TestUtil.generateData(spark);
        final GeneratedData data = TestUtil.prepareData(frame, PRNG_SEED, SAMPLE_SIZE);
        final ClassificationModelEvaluator.EvaluationResult result = generateMleResult(data, PRNG_SEED,
                LAYERS, null, SOLVER);

        PrintStream output = System.out;

        TestUtil.printHeader(output);
        TestUtil.printResults(result, output);

        Assertions.assertEquals(0.20244997100131956, result.getFittingEntropy().getCrossEntropy());
        Assertions.assertEquals(0.19955043889066668, result.getTestingEntropy().getCrossEntropy());
    }

    private ClassificationModelEvaluator.EvaluationResult generateMleResult(final TestUtil.GeneratedData data_,
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


}