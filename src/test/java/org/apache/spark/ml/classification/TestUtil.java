package org.apache.spark.ml.classification;

import edu.columbia.tjw.item.spark.ClassificationModelEvaluator;
import org.apache.parquet.Strings;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class TestUtil
{
    public static final List<String> INPUT_COLS = Collections
            .unmodifiableList(Arrays.asList("AGE", "MTM_LTV", "INCENTIVE",
                    "CREDIT_SCORE", "FIRSTTIME_BUYER", "UNIT_COUNT", "ORIG_CLTV", "ORIG_DTI", "UPB", "ORIG_INTRATE",
                    "PREPAYMENT_PENALTY"));

    public static final SparkSession generateSparkSession()
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
                new VectorAssembler().setInputCols(TestUtil.INPUT_COLS.toArray(new String[0])).setOutputCol("features")
                        .setHandleInvalid("skip");
        Dataset<Row> transformed = assembler.transform(dropped);
        return transformed;
    }


    public static void printHeader(final PrintStream output_)
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

    public static void printResults(final ClassificationModelEvaluator.EvaluationResult result_,
                                    final PrintStream output_)
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


    public static final GeneratedData prepareData(final Dataset<Row> raw_,
                                                  final long prngSeed_,
                                                  final int sampleSize_)
    {
        Dataset<Row> transformed = assemble(raw_);
        Dataset<Row>[] datasets = transformed.randomSplit(new double[]{0.25, 0.75}, prngSeed_);

        Dataset<Row> fitting = datasets[0].orderBy(functions.rand(prngSeed_)).limit(sampleSize_);
        Dataset<Row> testing = datasets[1];

        fitting.persist(StorageLevel.MEMORY_AND_DISK());
        return new GeneratedData(fitting, testing);
    }


    public static final class GeneratedData
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


    public static Dataset<Row> generateData(final SparkSession spark)
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
