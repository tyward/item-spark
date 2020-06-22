package org.apache.spark.ml.classification;

import edu.columbia.tjw.item.spark.ClassificationModelEvaluator;
import edu.columbia.tjw.item.util.MathTools;
import edu.columbia.tjw.item.util.random.PrngType;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.parquet.Strings;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.ann.IcePerceptronClassificationModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import org.junit.jupiter.api.Test;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class IcePerceptronClassifierTest
{
    private static final String[] INPUT_COLS = new String[]{"MTM_LTV", "INCENTIVE", "FIRSTTIME_BUYER",
            "UNIT_COUNT", "ORIG_CLTV", "ORIG_DTI", "ORIG_INTRATE", "PREPAYMENT_PENALTY"};
    //private static final String[] INPUT_COLS = new String[]{"MTM_LTV"};
    private static final int STATUS_COUNT = 3;


    private static final int[] LAYERS = new int[]{INPUT_COLS.length, 2, STATUS_COUNT};
    private static final int MAX_ITER = 500;
    //private static final int BLOCK_SIZE = 128;
    private static final int BLOCK_SIZE = 4 * 4096;
    private static final int SAMPLE_SIZE = BLOCK_SIZE;

    private static final String SOLVER = "l-bfgs";
    private static final String SOLVER_GD = "gd";
    private static final int PRNG_SEED = 12345;
    private static final double TOLERANCE = 1.0e-8;
    private static final double STEP_SIZE = 0.03;

//    @Test
//    void testICE()
//    {
//        final SparkSession spark = generateSparkSession();
//        final Dataset<Row> frame = generateData(spark);
//        final ClassificationModelEvaluator.EvaluationResult result = generateIceResult(frame, PRNG_SEED,
//                LAYERS, SAMPLE_SIZE, null, SOLVER);
//
//        PrintStream output = System.out;
//        printHeader(output);
//        printResults(result, output);
//    }
//
//    @Test
//    void testMLP()
//    {
//        final SparkSession spark = generateSparkSession();
//        final Dataset<Row> frame = generateData(spark);
//        final ClassificationModelEvaluator.EvaluationResult result = generateMleResult(frame, PRNG_SEED,
//                LAYERS, SAMPLE_SIZE, null, SOLVER);
//
//        PrintStream output = System.out;
//
//        printHeader(output);
//        printResults(result, output);
//    }

    @Test
    void testSweep() throws Exception
    {
        final SparkSession spark = generateSparkSession();
        final Dataset<Row> frame = generateData(spark);

        final Random rand = RandomTool.getRandom(PrngType.SECURE);

        final int[][] testLayers = new int[4][];
        testLayers[0] = new int[]{INPUT_COLS.length, STATUS_COUNT};
        testLayers[1] = new int[]{INPUT_COLS.length, 5, STATUS_COUNT};
        //testLayers[2] = new int[]{INPUT_COLS.length, 8, STATUS_COUNT};
        testLayers[2] = new int[]{INPUT_COLS.length, 5, 5, STATUS_COUNT};
        //testLayers[4] = new int[]{INPUT_COLS.length, 8, 5, STATUS_COUNT};
        //testLayers[5] = new int[]{INPUT_COLS.length, 8, 8, STATUS_COUNT};
        testLayers[3] = new int[]{INPUT_COLS.length, 8, 5, 5, STATUS_COUNT};

        final int[] sampleSizes = new int[]{128, 256, 512, 1024, 2048, 4096, 8 * 1024, 16 * 1024, 32 * 1024,
                64 * 1024, 128 * 1024};

        final int repCount = 10;

        try (final OutputStream oStream = new FileOutputStream("/Users/tyler/Desktop/runResults.csv");
             PrintStream output = new PrintStream(oStream))
        {
            printHeader(output);

            for (int k = 0; k < testLayers.length; k++)
            {
                for (int w = 0; w < sampleSizes.length; w++)
                {
                    for (int i = 0; i < repCount; i++)
                    {
                        final long prngSeed = rand.nextLong();

                        final ClassificationModelEvaluator.EvaluationResult startingPoint = generateMleResult(frame,
                                prngSeed, testLayers[k], sampleSizes[w], null, SOLVER_GD);
                        printResults(startingPoint, output);

                        {
                            final ClassificationModelEvaluator.EvaluationResult mlpResult = generateMleResult(frame,
                                    prngSeed, testLayers[k], sampleSizes[w], null, SOLVER);
                            printResults(mlpResult, output);
                        }

//                        {
//                            final ClassificationModelEvaluator.EvaluationResult mlpResult2 = generateMleResult(frame,
//                                    prngSeed, testLayers[k], sampleSizes[w], startingPoint.getModel().weights(),
//                                    SOLVER);
//                            printResults(mlpResult2, output);
//                        }

//                        {
//                            final ClassificationModelEvaluator.EvaluationResult mlpResult2 = generateMleResult(frame,
//                                    prngSeed, testLayers[k], sampleSizes[w], startingPoint.getModel().weights(),
//                                    SOLVER_GD);
//                            printResults(mlpResult2, output);
//                        }

                        // ICE after here.


                        {
                            final ClassificationModelEvaluator.EvaluationResult iceResult2 = generateIceResult(frame,
                                    prngSeed, testLayers[k], sampleSizes[w], null, SOLVER_GD);
                            printResults(iceResult2, output);
                        }

                        {
                            final ClassificationModelEvaluator.EvaluationResult iceResult = generateIceResult(frame,
                                    prngSeed, testLayers[k], sampleSizes[w], null, SOLVER);
                            printResults(iceResult, output);
                        }

                        {
                            final ClassificationModelEvaluator.EvaluationResult iceResult = generateIceResult(frame,
                                    prngSeed, testLayers[k], sampleSizes[w], startingPoint.getModel().weights(),
                                    SOLVER_GD);
                            printResults(iceResult, output);
                        }

                        {
                            final ClassificationModelEvaluator.EvaluationResult iceResult = generateIceResult(frame,
                                    prngSeed, testLayers[k], sampleSizes[w], startingPoint.getModel().weights(),
                                    SOLVER);
                            printResults(iceResult, output);
                        }


                    }
                }
            }
        }


//        final ClassificationModelEvaluator.EvaluationResult result = generateMlpResult(frame, PRNG_SEED,
//                LAYERS, SAMPLE_SIZE);
//
//        PrintStream output = System.out;
//
//        printHeader(output);
//        printResults(result, output);

    }


    private ClassificationModelEvaluator.EvaluationResult generateMleResult(Dataset<Row> raw, final long prngSeed_,
                                                                            final int[] layers_,
                                                                            final int sampleSize_,
                                                                            final Vector startingPoint_,
                                                                            final String solver_)
    {
        MultilayerPerceptronClassifier mlpFitter = new MultilayerPerceptronClassifier().setLabelCol("NEXT_STATUS");
        mlpFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(sampleSize_).setSolver(solver_).setTol(TOLERANCE)
                .setStepSize(STEP_SIZE);

        if (null != startingPoint_)
        {
            mlpFitter.setInitialWeights(startingPoint_);
        }

        final GeneratedData data = prepareData(raw, prngSeed_, sampleSize_);
        return ClassificationModelEvaluator
                .evaluate(mlpFitter, "MLE[" + (startingPoint_ == null) + "][" + solver_ + "]",
                        data.getFitting(), data.getTesting(),
                        prngSeed_,
                        layers_);
    }

    private ClassificationModelEvaluator.EvaluationResult generateIceResult(Dataset<Row> raw, final long prngSeed_,
                                                                            final int[] layers_, final int sampleSize_,
                                                                            final Vector startingPoint_,
                                                                            final String solver_)
    {
        IcePerceptronClassifier iceFitter = new IcePerceptronClassifier().setLabelCol("NEXT_STATUS");
        iceFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(sampleSize_).setSolver(solver_).setTol(TOLERANCE)
                .setStepSize(STEP_SIZE);

        if (null != startingPoint_)
        {
            iceFitter.setInitialWeights(startingPoint_);
        }

        final GeneratedData data = prepareData(raw, prngSeed_, sampleSize_);

        IcePerceptronClassificationModel model = (IcePerceptronClassificationModel) iceFitter.fit(data.getFitting());

        //validateGradients(iceFitter, data, model);

        return ClassificationModelEvaluator
                .evaluate(iceFitter,
                        "ICE[" + (startingPoint_ == null) + "][" + solver_ + "]", data.getFitting(),
                        data.getTesting(),
                        prngSeed_,
                        layers_);
    }

    private void validateGradients(IcePerceptronClassifier iceFitter, final GeneratedData data,
                                   IcePerceptronClassificationModel model)
    {
        final double[] weights = model.weights().toArray().clone();

        final int size = weights.length;

        final double[] grad = new double[size];
        final double[] jDiag = new double[size];

        final double[] fdGrad = new double[size];
        final double[] fdDiag = new double[size];

        final Dataset<Row> fittingData = data.getFitting().limit(1);


        final double loss = model.computeGradients(fittingData, model.weights(), grad, jDiag);

        for (int i = 0; i < weights.length; i++)
        {
            final double[] w2 = weights.clone();

            final double h = Math.max(Math.abs(w2[i] * 1.0e-4), 1.0e-12);
            w2[i] += h;

            final double[] g2 = new double[size];
            final double[] jDiag2 = new double[size];

            final double lossUp = model.computeGradients(fittingData, Vectors.dense(w2), g2, jDiag2);

            final double fdd = (lossUp - loss) / h;

            final double origGrad = grad[i];
            final double shiftGrad = g2[i];
            final double shiftDiag = jDiag2[i];
            final double origDiag = jDiag[i];

            final double fdd2 = (shiftGrad - origGrad) / h;

            fdGrad[i] = fdd;
            fdDiag[i] = fdd2;

            w2[i] = weights[i] - h;
            final double lossDown = model.computeGradients(fittingData, Vectors.dense(w2), g2, jDiag2);

            final double fdd2a = (lossUp + lossDown - (2 * loss)) / (h * h);
            final double fdd2b = (origGrad - g2[i]) / h;


            System.out.println("FDD[" + origGrad + "]: " + fdd);
            System.out.println("FDD2[" + origDiag + ", " + shiftDiag + "]: " + fdd2);
            System.out.println("next.");
        }

        final double fdCos = MathTools.cos(fdGrad, grad);
        final double fdCos2 = MathTools.cos(fdDiag, jDiag);

        final double magRatio = MathTools.magnitude(fdGrad) / MathTools.magnitude(grad);
        final double magRatio2 = MathTools.magnitude(fdDiag) / MathTools.magnitude(jDiag);

        final double[] fdGradA = new double[INPUT_COLS.length];
        final double[] fdDiagA = new double[INPUT_COLS.length];


        for (int i = 0; i < INPUT_COLS.length; i++)
        {
            // These are normalize, so the same H should work fine for everything.
            final double h = 1.0e-6;
//            final double scaleUp = 1.0 + h;
//            final double scaleDown = 1.0 - h;

            // Now update the contents of the row....
            final Dataset<Row> fdUp = assemble(fittingData.withColumn(INPUT_COLS[i],
                    functions.expr(INPUT_COLS[i] + " + " + h)));
            final Dataset<Row> fdDown = assemble(fittingData.withColumn(INPUT_COLS[i],
                    functions.expr(INPUT_COLS[i] + " - " + h)));

            final double[] g2 = new double[size];
            final double[] jDiag2 = new double[size];

            final double lossUp = model.computeGradients(fdUp, Vectors.dense(weights), g2, jDiag2);
            final double lossDown = model.computeGradients(fdDown, Vectors.dense(weights), g2, jDiag2);

            final double fdd = (lossUp - loss) / h;
            fdGradA[i] = fdd;

            final double fddDown = (loss - lossDown) / h;

            final double fdd2 = (lossUp + lossDown - (2 * loss)) / (h * h);
            fdDiagA[i] = fdd2;
        }

        final double[] grad2 = new double[size];
        final double[] jDiag2 = new double[size];
        final double loss2 = model.computeGradients(fittingData, model.weights(), grad2, jDiag2);

        System.out.println("Cos[" + magRatio + "]: " + fdCos);
        System.out.println("Cos2[" + magRatio2 + "]: " + fdCos2);
        System.out.println("Done.");

    }


    private void printHeader(final PrintStream output_)
    {
        final List<String> headers = new ArrayList<>();
        headers.add("label");
        headers.add("fitTime");
        headers.add("paramCount");
        headers.add("prngSeed");
        headers.add("layerString");

        headers.add("fitEvaltime");
        headers.add("fitRowcount");
        headers.add("fitEntropy");
        headers.add("fitDistEntropy");

        headers.add("testEvaltime");
        headers.add("testRowcount");
        headers.add("testEntropy");
        headers.add("testDistEntropy");

        final String outputLine = Strings.join(headers, ", ");
        output_.println(outputLine);
    }

    private void printResults(final ClassificationModelEvaluator.EvaluationResult result_, final PrintStream output_)
    {
        final ClassificationModelEvaluator.EntropyResult fitResult = result_.getFittingEntropy();
        final ClassificationModelEvaluator.EntropyResult testResult = result_.getTestingEntropy();

        final List<String> info = new ArrayList<>();
        info.add(result_.getLabel());
        info.add(Long.toString(result_.getFittingTime()));
        info.add(Integer.toString(result_.getParamCount()));
        info.add(Long.toHexString(result_.getPrngSeed()));
        info.add(result_.getLayerString().replace(", ", "|"));

        info.add(Long.toString(fitResult.getCalcTime()));
        info.add(Long.toString(fitResult.getRowCount()));
        info.add(Double.toString(fitResult.getCrossEntropy()));
        info.add(Double.toString(fitResult.getDistEntropy()));

        info.add(Long.toString(testResult.getCalcTime()));
        info.add(Long.toString(testResult.getRowCount()));
        info.add(Double.toString(testResult.getCrossEntropy()));
        info.add(Double.toString(testResult.getDistEntropy()));

        final String outputLine = Strings.join(info, ", ");
        output_.println(outputLine);
    }


    private static final SparkSession generateSparkSession()
    {
        final SparkSession spark = SparkSession.builder()
                .master("local")
                .appName("Tree Session")
                .getOrCreate();

        SparkContext context = spark.sparkContext();

        spark.udf().register("toArrayLambda", (x) -> (((DenseVector) x).toArray()),
                DataTypes.createArrayType(DataTypes.DoubleType));

        return spark;
    }

    private static final Dataset<Row> assemble(final Dataset<Row> input)
    {
        final Dataset<Row> dropped = input.drop("features");

        VectorAssembler assembler =
                new VectorAssembler().setInputCols(INPUT_COLS).setOutputCol("features").setHandleInvalid("skip");
        Dataset<Row> transformed = assembler.transform(dropped);
        return transformed;
    }

    private static final GeneratedData prepareData(final Dataset<Row> raw_, final long prngSeed_,
                                                   final int sampleSize_)
    {
        Dataset<Row> transformed = assemble(raw_);
        Dataset<Row>[] datasets = transformed.randomSplit(new double[]{0.25, 0.75}, prngSeed_);

        Dataset<Row> fitting = datasets[0].limit(sampleSize_);
        Dataset<Row> testing = datasets[1];

        fitting.persist(StorageLevel.MEMORY_AND_DISK());
//        testing.persist(StorageLevel.MEMORY_AND_DISK());

        return new GeneratedData(fitting, testing);
    }

    private static final class GeneratedData
    {
        private final Dataset<Row> _fitting;
        private final Dataset<Row> _testing;

        public GeneratedData(Dataset<Row> fitting_, Dataset<Row> testing_)
        {
            _fitting = fitting_;
            _testing = testing_;
        }

        public Dataset<Row> getFitting()
        {
            return _fitting;
        }

        public Dataset<Row> getTesting()
        {
            return _testing;
        }
    }


    private Dataset<Row> generateData(final SparkSession spark)
    {
        Dataset<Row> frame = spark.read().parquet("/Users/tyler/sync-workspace/nyu_class/data/df_c_1");

        frame = frame.filter("MTM_LTV > 0.0").filter("MTM_LTV < 2.0");

        frame = frame.filter("INCENTIVE > -1.0").filter("CREDIT_SCORE > 0");
        frame = frame.filter("NEXT_STATUS >= 0").filter("NEXT_STATUS <= 2");
        frame = frame.filter("AGE >= 0").filter("STATUS == 1");
        frame = frame.filter("TERM >= 0");
        frame = frame.filter("MI_PERCENT >= 0");
        frame = frame.filter("UNIT_COUNT >= 0");
        frame = frame.filter("ORIG_LTV >= 0");
        frame = frame.filter("ORIG_CLTV >= 0");
        frame = frame.filter("ORIG_DTI >= 0");
        frame = frame.filter("ORIG_UPB >= 0");
        frame = frame.filter("ORIG_INTRATE >= 0");

        // Go ahead and add the status columns here.
        frame = frame.withColumn("actual_p", functions.expr(" case when next_status = 0 then 1.0 else 0.0 end"));
        frame = frame.withColumn("actual_c", functions.expr(" case when next_status = 1 then 1.0 else 0.0 end"));
        frame = frame.withColumn("actual_3", functions.expr(" case when next_status = 2 then 1.0 else 0.0 end"));

        frame = frame.withColumn("FIRSTTIME_BUYER", functions.expr(" case when FIRSTTIME_BUYER then 1.0 else 0.0 end"));
        frame = frame.withColumn("PREPAYMENT_PENALTY",
                functions.expr(" case when PREPAYMENT_PENALTY then 1.0 else 0.0 end"));


        for (String nextCol : INPUT_COLS)
        {
            final double average =
                    extractDouble(frame.select(functions.expr("AVG(" + nextCol + ")")).toLocalIterator().next().get(0));
            final double dev =
                    extractDouble(
                            frame.select(functions.expr("STDDEV(" + nextCol + ")")).toLocalIterator().next().get(0));

            final double multiple;

            if (dev != 0)
            {
                multiple = 1.0 / dev;
            }
            else
            {
                multiple = 1.0;
            }

            frame = frame.withColumn(nextCol,
                    frame.col(nextCol).minus(functions.lit(average)).multiply(functions.lit(multiple)));

        }

        frame.persist(StorageLevel.MEMORY_AND_DISK());
        return frame;
    }

    private static double extractDouble(final Object o)
    {
        final Number num = (Number) o;
        return num.doubleValue();
    }

    private static double extractOne(DenseVector dv, int index)
    {
        return dv.apply(index);
    }
}