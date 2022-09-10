package org.apache.spark.ml.classification;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.ann.IcePerceptronClassificationModel;
import org.apache.spark.ml.classification.IcePerceptronClassifier;
import org.apache.spark.ml.classification.TestUtil.GeneratedData;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.Test;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class MnistTest
{
    private static final String FEATURES_COL = "features";
    private static final List<String> FEATURES;

    static
    {
        List<String> featureList = new ArrayList<>();

        for (int i = 0; i < 784; i++)
        {
            featureList.add("pixel" + i);
        }

        FEATURES = Collections.unmodifiableList(featureList);
    }


//    @Test
//    void testMNist()
//    {
//        final SparkSession spark = TestUtil.generateSparkSession();
//        final GeneratedData generated = generateData(spark);
//
//        final int[] layers = {FEATURES.size(), 10, 10};
//
//        final MultilayerPerceptronClassificationModel mlpModel;
//
//        {
//            MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
//                    .setLabelCol("label")
//                    .setFeaturesCol(FEATURES_COL)
//                    .setLayers(layers)
//                    .setSeed(42L)
//                    .setBlockSize(128) //default 128
//                    .setMaxIter(100) //default 100
//                    .setTol(1e-4); //default 1e-4
//
//
//            final long timestampStart = System.currentTimeMillis();
//            mlpModel = mlp.fit(generated.getFitting());
//            final long elapsed = System.currentTimeMillis() - timestampStart;
//        }
//
//        final Dataset<Row> results = mlpModel.transform(generated.getTesting());
//        evaluateResults(results);
//    }

    //@Test
    void testStored() throws IOException
    {
        final Pattern idPattern = Pattern.compile("ice_([\\d]+)_([\\d]+)_([\\d]+)");

//        Matcher match = idPattern.matcher("ice_0_800_0");
//        boolean isMatch = match.matches();
//
//
//        final int a = Integer.parseInt(match.group(1));
//        final int b = Integer.parseInt(match.group(3));


        final SparkSession spark = TestUtil.generateSparkSession();
        final GeneratedData generated = generateData(spark);

        System.out.println("Fitting count: " + generated.getFitting().count());
        System.out.println("Testing count: " + generated.getTesting().count());

        final int[] layers = {FEATURES.size(), 800, 10};
        final List<MNistEvaluation> evaluations = new ArrayList<>();

        final File mlpDir = new File("/Users/tyler/sync-workspace/math/research/mlpIce/mnist/models/mlp");
        mlpDir.mkdirs();

        for (int i = 0; i < 100; i++)
        {
            final File evalFile =
                    new File(mlpDir, "mlp_800b_eval_" + i);

            if (evalFile.exists())
            {
                evaluations.add(MNistEvaluation.load(evalFile));
                continue;
            }

            final File modelFile =
                    new File(mlpDir, "mlp_800b_" + i);

            if (!modelFile.exists())
            {
                break;
            }

            final MultilayerPerceptronClassificationModel mlpModel =
                    MultilayerPerceptronClassificationModel.load(modelFile.getAbsolutePath());

            final Dataset<Row> results = mlpModel.transform(generated.getTesting().withColumn(
                    "vecAssembler_025999af8334__output", functions.col("features")));

            final MNistEvaluation eval = evaluateResults("MLP[" + i + "]", layers, mlpModel.weights().size(), results);
            eval.save(evalFile);
            evaluations.add(eval);
        }

        final File iceDir = new File("/Users/tyler/sync-workspace/math/research/mlpIce/mnist/models/ice");
        iceDir.mkdirs();

        final File[] fileList = iceDir.listFiles();
        Arrays.sort(fileList);

        for (final File next : fileList)
        //for (int i = 0; i < 100; i++)
        {
            if (!next.isDirectory())
            {
                continue;
            }

            final File evalFile = new File(iceDir, next.getName() + ".eval");

            if (evalFile.exists())
            {
                evaluations.add(MNistEvaluation.load(evalFile));
                continue;
            }

            final String nextName = next.getName();
            final Matcher matcher = idPattern.matcher(nextName);

            if (!matcher.matches())
            {
                throw new IllegalStateException("Filename problems.");
            }

            final int j = Integer.parseInt(matcher.group(1));
            final int i = Integer.parseInt(matcher.group(3));

            final IcePerceptronClassificationModel mlpModel =
                    IcePerceptronClassificationModel.load(next.getAbsolutePath());

            final Dataset<Row> results = mlpModel.transform(generated.getTesting());

            final MNistEvaluation eval = evaluateResults("ICE[" + i + "|" + j + "]", layers, mlpModel.weights().size(),
                    results);
            eval.save(evalFile);
            evaluations.add(eval);
        }

        for (final MNistEvaluation eval : evaluations)
        {
            eval.print();
        }

        System.out.println("label, layers, paramCount, accuracy, crossEntropy, fMeasure");

        for (final MNistEvaluation eval : evaluations)
        {
            System.out.print(eval.getLabel());
            System.out.print(", ");
            System.out.print(eval.getLayerString());
            System.out.print(", ");
            System.out.print(eval.getParamCount());
            System.out.print(", ");
            System.out.print(eval.getAccuracy());
            System.out.print(", ");
            System.out.print(eval.getCrossEntropy());
            System.out.print(", ");
            System.out.println(eval.getWeightedFMeasure());
        }


//        {
//            final MultilayerPerceptronClassificationModel mlpModel = MultilayerPerceptronClassificationModel.load(
//                    "/Users/tyler/sync-workspace/math/research/mlpIce/mnist/mlp_800");
//            final Dataset<Row> results = mlpModel.transform(generated.getTesting().withColumn(
//                    "vecAssembler_e11e12e1576d__output", functions.col("features")));
//            evaluateResults(results).print();
//        }
//
//        {
//            final MultilayerPerceptronClassificationModel mlpModel = MultilayerPerceptronClassificationModel.load(
//                    "/Users/tyler/sync-workspace/math/research/mlpIce/mnist/mlp_800b");
//            final Dataset<Row> results = mlpModel.transform(generated.getTesting().withColumn(
//                    "vecAssembler_4104e8d5601c__output", functions.col("features")));
//            evaluateResults(results).print();
//        }
//
//        {
//            final IcePerceptronClassificationModel mlpModel = IcePerceptronClassificationModel.load(
//                    "/Users/tyler/sync-workspace/math/research/mlpIce/mnist/ice_800");
//            final Dataset<Row> results = mlpModel.transform(generated.getTesting().withColumn(
//                    "vecAssembler_e11e12e1576d__output", functions.col("features")));
//            evaluateResults(results).print();
//        }
//
//        {
//            final IcePerceptronClassificationModel mlpModel = IcePerceptronClassificationModel.load(
//                    "/Users/tyler/sync-workspace/math/research/mlpIce/mnist/ice_800b");
//            final Dataset<Row> results = mlpModel.transform(generated.getTesting().withColumn(
//                    "vecAssembler_4104e8d5601c__output", functions.col("features")));
//            evaluateResults(results).print();
//        }


//        final int[] layers = {FEATURES.size(), 10, 10};
//
//        final MultilayerPerceptronClassificationModel mlpModel;
//
//        {
//            MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
//                    .setLabelCol("label")
//                    .setFeaturesCol(FEATURES_COL)
//                    .setLayers(layers)
//                    .setSeed(42L)
//                    .setBlockSize(4 * 1024) //default 128
//                    .setMaxIter(100) //default 100
//                    .setTol(1e-4); //default 1e-4
//
//
//            final long timestampStart = System.currentTimeMillis();
//            mlpModel = mlp.fit(generated.getFitting());
//            final long elapsed = System.currentTimeMillis() - timestampStart;
//        }
//
//        final Dataset<Row> results = mlpModel.transform(generated.getTesting());
//        evaluateResults(results);
    }

    private static final class MNistEvaluation implements Serializable
    {
        private final String label;
        private final String layerString;
        private final int _paramCount;
        private final double accuracy;
        private final double crossEntropy;
        private final double weightedPrecision;
        private final double weightedRecall;
        private final double weightedFMeasure;
        private final double weightedFalsePositiveRate;
        private final Matrix confusionMatrix;

        public MNistEvaluation(final String label_, final int[] layers, final int paramCount, final Dataset<Row> result)
        {
            label = label_;
            layerString = Arrays.toString(layers).replace(',', '|');
            _paramCount = paramCount;

            Dataset<Row> mlp_pred =
                    result.select(functions.col("prediction"), functions.col("label"), functions.lit(1.0),
                            functions.col("probability")).withColumnRenamed("1.0", "weight");

            mlp_pred = mlp_pred.withColumn("assignedProbability", functions.expr("extractIndex(probability, label)"));

            //mlp_pred.show();

            JavaRDD<Row> converted = mlp_pred.javaRDD();
            JavaRDD<Tuple2<Object, Object>> mapped = converted.map(new MapFunction());


            MulticlassMetrics metrics = new MulticlassMetrics(JavaRDD.toRDD(mapped));

            accuracy = metrics.accuracy();
            crossEntropy =
                    mlp_pred.select(functions.avg(functions.expr("-1.0 * log(assignedProbability)"))).toLocalIterator()
                            .next().getDouble(0);
            weightedPrecision = metrics.weightedPrecision();
            weightedRecall = metrics.weightedRecall();
            weightedFMeasure = metrics.weightedFMeasure();
            weightedFalsePositiveRate = metrics.weightedFalsePositiveRate();

            confusionMatrix = metrics.confusionMatrix();
        }

        public void save(final File outputFile_) throws IOException
        {
            try (final FileOutputStream fout = new FileOutputStream(outputFile_);
                 final ObjectOutputStream oOut = new ObjectOutputStream(fout))
            {
                oOut.writeObject(this);
                oOut.flush();
            }
        }

        public static MNistEvaluation load(final File inputFile_) throws IOException
        {
            try (final FileInputStream fin = new FileInputStream(inputFile_);
                 final ObjectInputStream oIn = new ObjectInputStream(fin))
            {
                return (MNistEvaluation) oIn.readObject();
            }
            catch (final ClassNotFoundException e)
            {
                throw new IOException(e);
            }
        }


        public void print()
        {
            System.out.println("Label: " + label);
            System.out.println("Layers: " + layerString.replace(",", "|"));
            System.out.println("Param Count: " + _paramCount);
            System.out.println("Confusion matrix:");
            System.out.println(confusionMatrix);

            // Overall Statistics
            System.out.println("Summary Statistics");
            System.out.println("Accuracy = " + accuracy);
            System.out.println("Cross Entropy = " + crossEntropy);

            // Weighted stats
            System.out.println("Weighted precision: " + weightedPrecision);
            System.out.println("Weighted recall: " + weightedRecall);
            System.out.println("Weighted F1 score: " + weightedFMeasure);
            System.out.println("Weighted false positive rate: " + weightedFalsePositiveRate);
        }

        public String getLabel()
        {
            return label;
        }

        public String getLayerString()
        {
            return layerString;
        }

        public int getParamCount()
        {
            return _paramCount;
        }

        public double getAccuracy()
        {
            return accuracy;
        }

        public double getCrossEntropy()
        {
            return crossEntropy;
        }

        public double getWeightedPrecision()
        {
            return weightedPrecision;
        }

        public double getWeightedRecall()
        {
            return weightedRecall;
        }

        public double getWeightedFMeasure()
        {
            return weightedFMeasure;
        }

        public double getWeightedFalsePositiveRate()
        {
            return weightedFalsePositiveRate;
        }

        public Matrix getConfusionMatrix()
        {
            return confusionMatrix;
        }
    }


    public static MNistEvaluation evaluateResults(final String label_, final int[] layers,
                                                  final int paramCount, final Dataset<Row> result)
    {
        return new MNistEvaluation(label_, layers, paramCount, result);
    }


    public static GeneratedData generateData(final SparkSession spark)
    {

        final Dataset<Row> test = spark.read().parquet("/Users/tyler/sync-workspace/math/research/mlpIce/mnist" +
                "/mnist_test");
        final Dataset<Row> train = spark.read()
                .parquet("/Users/tyler/sync-workspace/math/research/mlpIce/mnist/mnist_train");

        final String[] features = FEATURES.toArray(new String[FEATURES.size()]);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(features).setOutputCol(FEATURES_COL);


        Dataset<Row> trainAssembled = assembler.transform(train);
        Dataset<Row> testAssembled = assembler.transform(test);

        trainAssembled = trainAssembled.persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY());
        testAssembled = testAssembled.persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY());

        return new GeneratedData(trainAssembled, testAssembled);

    }


    private static final class MapFunction implements Function<Row, Tuple2<Object, Object>>, Serializable
    {
        @Override
        public Tuple2<Object, Object> call(Row o) throws Exception
        {
            return new Tuple2<Object, Object>(o.getDouble(0), (double) o.getInt(1));
        }
    }
}
