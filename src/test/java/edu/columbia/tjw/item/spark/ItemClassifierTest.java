package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.util.random.PrngType;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

public class ItemClassifierTest
{

    @Test
    public void testBasicClassifier()
    {
        final SparkSession spark = SparkSession.builder()
                .master("local")
                .appName("Tree Session")
                .getOrCreate();

        SparkContext context = spark.sparkContext();

        spark.udf().register("toArrayLambda", (x) -> (((DenseVector) x).toArray()),
                DataTypes.createArrayType(DataTypes.DoubleType));

        Dataset<Row> frame = generateData(spark);

        List<String> inputCols = Arrays.asList("MTM_LTV", "INCENTIVE", "FIRSTTIME_BUYER",
                "UNIT_COUNT", "ORIG_CLTV", "ORIG_DTI", "ORIG_INTRATE", "PREPAYMENT_PENALTY");

        SortedSet<String> curveColumns = new TreeSet<>(inputCols);

        final ItemSettings.Builder settingsBuilder = ItemSettings.newBuilder();
        settingsBuilder.setRand(RandomTool.getRandom(PrngType.SECURE, 12345));

        ItemClassifierSettings settings = ItemClassifier.prepareSettings(frame, "NEXT_STATUS", inputCols,
                curveColumns,
                10, settingsBuilder.build());

        Dataset<Row> itemAssembled = ItemClassifier.prepareData(frame, settings, "features");
        Dataset<Row>[] datasets = itemAssembled.randomSplit(new double[]{0.25, 0.75}, 12345);

        Dataset<Row> fitting = datasets[0].limit(10 * 1000);
        Dataset<Row> testing = datasets[1];

        final ItemClassificationModel itemModel;


        {
            ItemClassifier classifier = new ItemClassifier(settings).setLabelCol("NEXT_STATUS").setFeaturesCol(
                    "features");

            itemModel = classifier.fit(fitting);
        }

        Evaluation fitEvalItem = evaluate(spark, testing, itemModel);

        Assertions.assertEquals(0.21147478070722345, fitEvalItem.getCrossEntropy());
        Assertions.assertEquals(0.19139604117045084, fitEvalItem.getDistEntropy());
        Assertions.assertEquals(1471696, fitEvalItem.getRowCount());
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

    private static final class Evaluation
    {
        private final long _rowCount;
        private final double _crossEntropy;
        private final double _distEntropy;


        public Evaluation(final long rowCount_, final double crossEntropy_, final double distEntropy_)
        {
            _rowCount = rowCount_;
            _crossEntropy = crossEntropy_;
            _distEntropy = distEntropy_;
        }

        public long getRowCount()
        {
            return _rowCount;
        }

        public double getCrossEntropy()
        {
            return _crossEntropy;
        }

        public double getDistEntropy()
        {
            return _distEntropy;
        }
    }


    private Evaluation evaluate(final SparkSession spark, Dataset<Row> data,
                                ClassificationModel model)
    {
        Dataset<Row> testingResults = model.transform(data);
        testingResults = testingResults.withColumn("prob_array", functions.expr("toArrayLambda(probability)"));
        testingResults = testingResults.withColumn("prob_p", functions.expr("prob_array[0]"));
        testingResults = testingResults.withColumn("prob_c", functions.expr("prob_array[1]"));
        testingResults = testingResults.withColumn("prob_3", functions.expr("prob_array[2]"));

        testingResults = testingResults.withColumn("distEntropy", functions.expr("-1.0 * ((prob_p*log(prob_p))  + " +
                "(prob_c*log(prob_c)) + (prob_3*log(prob_3)))"));
        testingResults = testingResults
                .withColumn("crossEntropy", functions.expr(" -1.0 * log(prob_array[next_status])"));

        testingResults.show();

        Dataset<Row> result = testingResults.select(functions.expr("count(*)"), functions.expr("sum" +
                        "(crossEntropy)/count(*)"),
                functions.expr(
                        "sum(distEntropy)/count(*)"));


        final Row fitRow = result.toLocalIterator().next();
        final long fitCount = fitRow.getLong(0);
        final double fitEntropy = fitRow.getDouble(1);
        final double fitDistEntropy = fitRow.getDouble(2);

        final Evaluation output = new Evaluation(fitCount, fitEntropy, fitDistEntropy);

        result.show();

        return output;
    }


    private static double extractOne(DenseVector dv, int index)
    {
        return dv.apply(index);
    }
}
