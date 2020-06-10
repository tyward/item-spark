package org.apache.spark.ml.classification;

import edu.columbia.tjw.item.spark.ClassificationModelEvaluator;
import edu.columbia.tjw.item.util.random.PrngType;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.parquet.Strings;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
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
    private static final int STATUS_COUNT = 3;


    private static final int[] LAYERS = new int[]{INPUT_COLS.length, 5, STATUS_COUNT};
    private static final int MAX_ITER = 100;
    //private static final int BLOCK_SIZE = 128;
    private static final int BLOCK_SIZE = 4 * 4096;
    private static final int SAMPLE_SIZE = BLOCK_SIZE;

    private static final String SOLVER = "l-bfgs";
    private static final int PRNG_SEED = 12345;

    @Test
    void testICE()
    {
        final SparkSession spark = generateSparkSession();
        final Dataset<Row> frame = generateData(spark);
        final ClassificationModelEvaluator.EvaluationResult result = generateIceResult(frame, PRNG_SEED,
                LAYERS, SAMPLE_SIZE);

        PrintStream output = System.out;
        printHeader(output);
        printResults(result, output);
    }

    @Test
    void testMLP()
    {
        final SparkSession spark = generateSparkSession();
        final Dataset<Row> frame = generateData(spark);
        final ClassificationModelEvaluator.EvaluationResult result = generateMlpResult(frame, PRNG_SEED,
                LAYERS, SAMPLE_SIZE);

        PrintStream output = System.out;

        printHeader(output);
        printResults(result, output);
    }

    @Test
    void testSweep() throws Exception
    {
        final SparkSession spark = generateSparkSession();
        final Dataset<Row> frame = generateData(spark);

        final Random rand = RandomTool.getRandom(PrngType.SECURE);

        final int[][] testLayers = new int[5][];
        testLayers[0] = new int[]{INPUT_COLS.length, STATUS_COUNT};
        testLayers[1] = new int[]{INPUT_COLS.length, 5, STATUS_COUNT};
        testLayers[2] = new int[]{INPUT_COLS.length, 5, 5, STATUS_COUNT};
        testLayers[3] = new int[]{INPUT_COLS.length, 8, 5, STATUS_COUNT};
        testLayers[4] = new int[]{INPUT_COLS.length, 8, 5, 5, STATUS_COUNT};

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
                        final ClassificationModelEvaluator.EvaluationResult mlpResult = generateMlpResult(frame,
                                prngSeed, testLayers[k], sampleSizes[w]);
                        printResults(mlpResult, output);

                        final ClassificationModelEvaluator.EvaluationResult iceResult = generateIceResult(frame,
                                prngSeed, testLayers[k], sampleSizes[w]);
                        printResults(iceResult, output);
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


    private ClassificationModelEvaluator.EvaluationResult generateMlpResult(Dataset<Row> raw, final long prngSeed_,
                                                                            final int[] layers_, final int sampleSize_)
    {
        MultilayerPerceptronClassifier mlpFitter = new MultilayerPerceptronClassifier().setLabelCol("NEXT_STATUS");
        mlpFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(sampleSize_).setSolver(SOLVER);

        final GeneratedData data = prepareData(raw, prngSeed_, sampleSize_);
        return ClassificationModelEvaluator.evaluate(mlpFitter, "MLP", data.getFitting(), data.getTesting(), prngSeed_,
                layers_);
    }

    private ClassificationModelEvaluator.EvaluationResult generateIceResult(Dataset<Row> raw, final long prngSeed_,
                                                                            final int[] layers_, final int sampleSize_)
    {
        IcePerceptronClassifier iceFitter = new IcePerceptronClassifier().setLabelCol("NEXT_STATUS");
        iceFitter.setLayers(layers_).setSeed(prngSeed_).setLabelCol("NEXT_STATUS")
                .setMaxIter(MAX_ITER).setBlockSize(sampleSize_).setSolver(SOLVER);

        final GeneratedData data = prepareData(raw, prngSeed_, sampleSize_);
        return ClassificationModelEvaluator.evaluate(iceFitter, "ICE", data.getFitting(), data.getTesting(), prngSeed_,
                layers_);
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

    private static final GeneratedData prepareData(final Dataset<Row> raw_, final long prngSeed_,
                                                   final int sampleSize_)
    {
        VectorAssembler assembler =
                new VectorAssembler().setInputCols(INPUT_COLS).setOutputCol("features").setHandleInvalid("skip");
        Dataset<Row> transformed = assembler.transform(raw_);
        Dataset<Row>[] datasets = transformed.randomSplit(new double[]{0.25, 0.75}, prngSeed_);

        Dataset<Row> fitting = datasets[0].limit(sampleSize_);
        Dataset<Row> testing = datasets[1];

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
        return frame;
    }


    private static double extractOne(DenseVector dv, int index)
    {
        return dv.apply(index);
    }
}