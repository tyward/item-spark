package edu.columbia.tjw.item.spark;

import org.apache.spark.SparkContext;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
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


//        String[] inputCols = new String[]{"AGE", "MTM_LTV", "INCENTIVE", "CREDIT_SCORE", "FIRSTTIME_BUYER", "TERM",
//                "MI_PERCENT", "UNIT_COUNT", "ORIG_CLTV", "ORIG_DTI", "ORIG_UPB", "ORIG_INTRATE",
//                "PREPAYMENT_PENALTY"};

        List<String> inputCols = Arrays.asList("MTM_LTV", "INCENTIVE", "FIRSTTIME_BUYER",
                "UNIT_COUNT", "ORIG_CLTV", "ORIG_DTI", "ORIG_INTRATE", "PREPAYMENT_PENALTY");

        SortedSet<String> curveColumns = new TreeSet<>(inputCols);

        ItemClassifierSettings settings = ItemClassifier.prepareSettings(frame, "NEXT_STATUS", inputCols,
                curveColumns,
                10);

        Dataset<Row> itemAssembled = ItemClassifier.prepareData(frame, settings, "features");
        Dataset<Row>[] datasets = itemAssembled.randomSplit(new double[]{0.25, 0.75}, 12345);

        Dataset<Row> fitting = datasets[0].limit(10 * 1000);
        Dataset<Row> testing = datasets[1];

        final ClassificationModel itemModel;


        {
            ItemClassifier classifier = new ItemClassifier(settings).setLabelCol("NEXT_STATUS").setFeaturesCol(
                    "features");

            itemModel = classifier.fit(fitting);
        }


        Dataset<Row> fitEvalItem = evaluate(spark, testing, itemModel);

        System.out.println("Testing eval ICE.");
        fitEvalItem.show();
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


    private Dataset<Row> evaluate(final SparkSession spark, Dataset<Row> data,
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

        result.show();

        return result;
    }


    private static double extractOne(DenseVector dv, int index)
    {
        return dv.apply(index);
    }
}