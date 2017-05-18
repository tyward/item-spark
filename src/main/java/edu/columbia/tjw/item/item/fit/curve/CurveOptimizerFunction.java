/*
 * Copyright 2014 Tyler Ward.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * This code is part of the reference implementation of http://arxiv.org/abs/1409.6075
 * 
 * This is provided as an example to help in the understanding of the ITEM model system.
 */
package edu.columbia.tjw.item.fit.curve;

import edu.columbia.tjw.item.ItemCurve;
import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveParams;
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.data.RandomizedStatusGrid.MappedReader;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.util.RectangularDoubleArray;
import edu.columbia.tjw.item.util.LogLikelihood;
import edu.columbia.tjw.item.util.MultiLogistic;
import edu.columbia.tjw.item.optimize.EvaluationResult;
import edu.columbia.tjw.item.optimize.MultivariateDifferentiableFunction;
import edu.columbia.tjw.item.optimize.MultivariateGradient;
import edu.columbia.tjw.item.optimize.MultivariatePoint;
import edu.columbia.tjw.item.optimize.ThreadedMultivariateFunction;
import java.util.List;

/**
 *
 * @author tyler
 * @param <S> The status type for this optimizer function
 * @param <R> The regressor type for this optimizer function
 * @param <T> THe curve type for this optimizer function
 */
public class CurveOptimizerFunction<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
        extends ThreadedMultivariateFunction implements MultivariateDifferentiableFunction
{
    private final LogLikelihood<S> _likelihood;
    private final ItemCurveFactory<R, T> _factory;
    private final int _size;
    private final double[] _workspace;
    private final int _toIndex;
    private final int[] _indexList;

    private final CurveParamsFitter<S, R, T> _curveFitter;

    private final S _status;
    private final int[] _actualOffsets;

    private MultivariatePoint _prevPoint;

    private final ItemSettings _settings;

    private final ItemCurveParams<R, T> _initParams;
    private final boolean _subtractStarting;

    private ItemCurveParams<R, T> _params;

    //N.B: This is an unsafe reference to an array owned by someone else, be careful with it.
    private final float[][] _regData;

    public CurveOptimizerFunction(final ItemCurveParams<R, T> initParams_, final ItemCurveFactory<R, T> factory_, final S fromStatus_, final S toStatus_, final CurveParamsFitter<S, R, T> curveFitter_,
            final int[] actualOrdinals_, final ParamFittingGrid<S, R, T> grid_, final int[] indexList_, final ItemSettings settings_, final boolean subtractStarting_)
    {
        super(settings_.getThreadBlockSize(), settings_.getUseThreading());

        _indexList = indexList_;
        _factory = factory_;
        _initParams = initParams_;
        _subtractStarting = subtractStarting_;

        _curveFitter = curveFitter_;
        _settings = settings_;
        _likelihood = new LogLikelihood<>(fromStatus_);
        _actualOffsets = new int[actualOrdinals_.length];

        //Convert ordinals to offsets. 
        for (int i = 0; i < _actualOffsets.length; i++)
        {
            _actualOffsets[i] = _likelihood.ordinalToOffset(actualOrdinals_[i]);
        }

        _status = fromStatus_;
        _size = _actualOffsets.length;
        _workspace = new double[_initParams.size()];
        _toIndex = fromStatus_.getReachable().indexOf(toStatus_);

        final int depth = _initParams.getEntryDepth();
        _regData = new float[depth][];

        for (int i = 0; i < depth; i++)
        {
            final R reg = _initParams.getRegressor(i);
            //_readers[i] = grid_.getRegressorReader(reg);

            final MappedReader reader = (MappedReader) grid_.getRegressorReader(reg);

            _regData[i] = reader.getUnderlyingArray();
        }
    }

    @Override
    public int dimension()
    {
        return _initParams.size();
    }

    @Override
    public int numRows()
    {
        return _size;
    }

    @Override
    protected void prepare(MultivariatePoint input_)
    {
        if (null != _prevPoint)
        {
            if (_prevPoint.equals(input_))
            {
                return;
            }
        }

        _prevPoint = input_.clone();

        for (int i = 0; i < input_.getDimension(); i++)
        {
            _workspace[i] = input_.getElement(i);
        }

        _params = new ItemCurveParams<>(_initParams, _factory, _workspace);
    }

    @Override
    protected void evaluate(int start_, int end_, EvaluationResult result_)
    {
        if (start_ == end_)
        {
            return;
        }

        final RectangularDoubleArray powerScores = _curveFitter.getPowerScores();
        final int cols = powerScores.getColumns();

        final double[] computed = new double[cols];
        //final double[] actual = new double[cols];

        final int depth = _params.getEntryDepth();

        // N.B: The intercept adjustment from the prev params has already been absorbed into 
        // the intercept term. No need to redo it or adjust for it here, it's already part of 
        // the baseline, we are fitting only an additive adjustment on top of that.
        final double interceptAdjustment = _params.getIntercept();
        final double beta = _params.getBeta();
        final double prevBeta = _initParams.getBeta();

        for (int i = start_; i < end_; i++)
        {
            for (int k = 0; k < cols; k++)
            {
                computed[k] = powerScores.get(i, k);
                //actual[k] = _actualProbabilities.get(i, k);
            }

            final int actualOffset = _actualOffsets[i];

            final int mapped = _indexList[i];
            double weight = 1.0;

            for (int k = 0; k < depth; k++)
            {
                final double regressor = _regData[k][mapped];
                final ItemCurve<T> trans = _params.getCurve(k);
                final double transformed;

                if (null == trans)
                {
                    transformed = regressor;
                }
                else
                {
                    transformed = trans.transform(regressor);
                }

                weight *= transformed;
            }

            double contribution = beta * weight;

//            final double regressor = _regData[0][mapped];
//
//            final double transformed = trans.transform(regressor);
//            final double contribution = (beta * transformed);
//            final double prevContribution;
            if (_subtractStarting)
            {
                double prevWeight = 1.0;

                for (int k = 0; k < depth; k++)
                {
                    final double regressor = _regData[k][mapped];
                    final ItemCurve<T> trans = _initParams.getCurve(k);
                    final double transformed;

                    if (null == trans)
                    {
                        transformed = regressor;
                    }
                    else
                    {
                        transformed = trans.transform(regressor);
                    }

                    prevWeight *= transformed;
                }

                contribution -= prevBeta * prevWeight;
            }

            //We are replacing one curve with another (if _prevCurve != null), so subtract off the 
            // curve we previously had before adding this new one.
            final double totalContribution = interceptAdjustment + contribution;

            computed[_toIndex] += totalContribution;

            //Converte these power scores into probabilities.
            MultiLogistic.multiLogisticFunction(computed, computed);

            final double logLikelihood = _likelihood.logLikelihood(computed, actualOffset);

            result_.add(logLikelihood, result_.getHighWater(), i + 1);
        }
    }

    @Override
    protected MultivariateGradient evaluateDerivative(int start_, int end_, MultivariatePoint input_, EvaluationResult result_)
    {
        final int dimension = input_.getDimension();
        final double[] derivative = new double[dimension];

        if (start_ >= end_)
        {
            final MultivariatePoint der = new MultivariatePoint(derivative);
            return new MultivariateGradient(input_, der, null, 0.0);
        }

        final List<S> reachable = _status.getReachable();
        int count = 0;
        final int reachableCount = reachable.size();

        final double[] scores = new double[reachableCount];

        //We are only interested in the specific element being curved....
        //Therefore, set the beta to 1.0, the result is a multiple of beta
        //for the special case where only one beta is set. We will scale afterwards. 
        final double[] betas = new double[reachableCount];
        betas[this._toIndex] = 1.0;

        final double[] workspace1 = new double[reachableCount];
        final double[] output = new double[reachableCount];
        //final double[] actual = new double[reachableCount];

        final int depth = _params.getEntryDepth();
        final double[] weights = new double[depth];
        final double[] knockoutWeights = new double[depth];
        //final double[] componentDeriv = new double[depth];

        final int interceptIndex = _params.getInterceptIndex();
        final int betaIndex = _params.getBetaIndex();

        final RectangularDoubleArray powerScores = _curveFitter.getPowerScores();

        //N.B: The derivative depes only on current params, not on _initParms, regardless of _subtractStarting.
        final double beta = _params.getBeta();

        for (int i = start_; i < end_; i++)
        {
            for (int w = 0; w < reachableCount; w++)
            {
                scores[w] = powerScores.get(i, w);
                //actual[w] = _actualProbabilities.get(i, w);
            }

            //After this, workspace1 holds the model probabilities, output holds the xDerivatives of the probabilities.
            MultiLogistic.multiLogisticRegressorDerivatives(scores, betas, workspace1, output);
            MultiLogistic.multiLogisticFunction(scores, workspace1);

            double xDerivative = 0.0;

            final int actualOffset = _actualOffsets[i];

            if (actualOffset >= 0)
            {
                //N.B: Ignore any transitions that we know to be impossible.
                final double derivTerm = output[actualOffset] / workspace1[actualOffset];
                xDerivative += derivTerm;
            }

            final int mapped = _indexList[i];
            double weight = 1.0;

            for (int k = 0; k < depth; k++)
            {
                final double regressor = _regData[k][mapped];
                final ItemCurve<T> trans = _params.getCurve(k);
                final double transformed;

                if (null == trans)
                {
                    transformed = regressor;
                }
                else
                {
                    transformed = trans.transform(regressor);
                }

                weights[k] = transformed;
                knockoutWeights[k] = 1.0;
                weight *= transformed;
            }

            //Fill in the weight of everything but this curve....
            for (int k = 0; k < depth; k++)
            {
                for (int w = 0; w < depth; w++)
                {
                    if (w == k)
                    {
                        continue;
                    }

                    knockoutWeights[k] *= weights[w];
                }
            }

            //In our special case, the derivative is directly proportional to beta, because we apply it to only one state. 
            final double interceptDerivative = xDerivative;

            final double betaDerivative = xDerivative * weight;

            derivative[interceptIndex] += interceptDerivative;
            derivative[betaIndex] += betaDerivative;

            for (int w = 0; w < _params.size(); w++)
            {
                if (w == interceptIndex || w == betaIndex)
                {
                    //Already handled these cases.
                    continue;
                }

                final int depthIndex = _params.indexToCurveIndex(w);
                final double regressor = _regData[depthIndex][mapped];
                final ItemCurve<T> targetCurve = _params.getCurve(depthIndex);
                final int curveOffset = _params.indexToCurveOffset(w);
                final double paramDerivative = targetCurve.derivative(curveOffset, regressor);
                final double knockoutWeight = knockoutWeights[depthIndex];

                final double deriv = xDerivative * beta * paramDerivative * knockoutWeight;
                derivative[w] += deriv;
            }

//            for (int w = 0; w < trans.getCurveType().getParamCount(); w++)
//            {
//                final double paramDerivative = trans.derivative(w, regressor);
//                final double combined = xDerivative * beta * paramDerivative;
//                derivative[w] += combined;
//            }
            count++;
        }

        //N.B: we are computing the negative log likelihood. 
        final double invCount = -1.0 / count;

        for (int i = 0; i < derivative.length; i++)
        {
            derivative[i] = derivative[i] * invCount;
        }

        final MultivariatePoint der = new MultivariatePoint(derivative);

        final MultivariateGradient grad = new MultivariateGradient(input_, der, null, 0.0);
        return grad;
    }

    @Override
    public int resultSize(int start_, int end_)
    {
        final int size = (end_ - start_);
        return size;
    }

}
