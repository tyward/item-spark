package edu.columbia.tjw.item.spark;

import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.classification.ProbabilisticClassifier;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.util.Arrays;

public class ClassificationModelEvaluator
{


    public static <W extends ProbabilisticClassificationModel<Vector, W>, M extends ProbabilisticClassifier<Vector, M
            , W>>
    EvaluationResult evaluate(final M classifier_,
                              final String label_,
                              Dataset<Row> fitting,
                              Dataset<Row> testing, final long prngSeed_, final int[] layers_)
    {
        final long start = System.currentTimeMillis();
        final MultilayerPerceptronClassificationModel evalModel = (MultilayerPerceptronClassificationModel) classifier_
                .fit(fitting);
        final long elapsed = System.currentTimeMillis() - start;

        final EntropyResult fitResult = computeEntropy(fitting, evalModel);
        final EntropyResult testResult = computeEntropy(testing, evalModel);

        final EvaluationResult result = new EvaluationResult(label_, prngSeed_, Arrays.toString(layers_), evalModel,
                elapsed,
                fitResult, testResult);
        return result;
    }


    private static EntropyResult computeEntropy(Dataset<Row> data,
                                                ClassificationModel model)
    {
        final long start = System.currentTimeMillis();
        Dataset<Row> testingResults = model.transform(data);
        testingResults = testingResults.withColumn("prob_array", functions.expr("toArrayLambda(probability)"));
        testingResults = testingResults.withColumn("prob_p", functions.expr("prob_array[0]"));
        testingResults = testingResults.withColumn("prob_c", functions.expr("prob_array[1]"));
        testingResults = testingResults.withColumn("prob_3", functions.expr("prob_array[2]"));

        testingResults = testingResults.withColumn("distEntropy", functions.expr("-1.0 * ((prob_p*log(prob_p))  + " +
                "(prob_c*log(prob_c)) + (prob_3*log(prob_3)))"));
        testingResults = testingResults
                .withColumn("crossEntropy", functions.expr(" -1.0 * log(prob_array[next_status])"));

        Dataset<Row> result = testingResults.select(functions.expr("count(*)"), functions.expr("sum" +
                        "(crossEntropy)/count(*)"),
                functions.expr(
                        "sum(distEntropy)/count(*)"));

        final Row fitRow = result.toLocalIterator().next();
        final long fitCount = fitRow.getLong(0);
        final double fitEntropy = fitRow.getDouble(1);
        final double fitDistEntropy = fitRow.getDouble(2);
        final long elapsed = System.currentTimeMillis() - start;

        final EntropyResult output = new EntropyResult(elapsed, fitCount, fitEntropy, fitDistEntropy);
        return output;
    }

    public static final class EntropyResult
    {
        private final long _calcTime;
        private final long _rowCount;
        private final double _crossEntropy;
        private final double _distEntropy;

        public EntropyResult(final long calcTime_, final long rowCount_, final double crossEntropy_,
                             final double distEntropy_)
        {
            _calcTime = calcTime_;
            _rowCount = rowCount_;
            _crossEntropy = crossEntropy_;
            _distEntropy = distEntropy_;
        }

        public long getCalcTime()
        {
            return _calcTime;
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


    public static final class EvaluationResult
    {
        private final String _label;
        private final long _prngSeed;
        private final String _layerString;
        private final MultilayerPerceptronClassificationModel _model;
        private final long _fittingTime;
        private final EntropyResult _fittingEntropy;
        private final EntropyResult _testingEntropy;

        public EvaluationResult(final String label_, long prngSeed_,
                                String layerString_, MultilayerPerceptronClassificationModel model_,
                                final long fittingTime_,
                                final EntropyResult fittingEntropy_, final EntropyResult testingEntropy_)
        {
            _label = label_;
            _prngSeed = prngSeed_;
            _layerString = layerString_;

            _model = model_;
            _fittingTime = fittingTime_;

            _fittingEntropy = fittingEntropy_;
            _testingEntropy = testingEntropy_;
        }


        public String getLabel()
        {
            return _label;
        }

        public MultilayerPerceptronClassificationModel getModel()
        {
            return _model;
        }

        public long getFittingTime()
        {
            return _fittingTime;
        }

        public EntropyResult getFittingEntropy()
        {
            return _fittingEntropy;
        }

        public EntropyResult getTestingEntropy()
        {
            return _testingEntropy;
        }

        public int getParamCount()
        {
            return _model.weights().size();
        }

        public long getPrngSeed()
        {
            return _prngSeed;
        }

        public String getLayerString()
        {
            return _layerString;
        }
    }
}
