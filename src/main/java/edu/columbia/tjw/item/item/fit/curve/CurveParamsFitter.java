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

import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveParams;
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.algo.QuantileDistribution;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.fit.EntropyCalculator;
import edu.columbia.tjw.item.fit.FittingProgressChain;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.optimize.MultivariateOptimizer;
import edu.columbia.tjw.item.optimize.MultivariatePoint;
import edu.columbia.tjw.item.optimize.OptimizationResult;
import edu.columbia.tjw.item.util.LogUtil;
import edu.columbia.tjw.item.util.MathFunctions;
import edu.columbia.tjw.item.util.MultiLogistic;
import edu.columbia.tjw.item.util.RectangularDoubleArray;
import java.util.Arrays;
import java.util.logging.Logger;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public final class CurveParamsFitter<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private static final Logger LOG = LogUtil.getLogger(CurveParamsFitter.class);

    private final ItemCurveFactory<R, T> _factory;
    private final ItemSettings _settings;

    private final ItemStatusGrid<S, R> _grid;
    private final RectangularDoubleArray _powerScores;

    // Getting pretty messy, we can now reset our grids and models, recache the power scores, etc...
    private final ParamFittingGrid<S, R, T> _paramGrid;
    private final ItemModel<S, R, T> _model;

    //N.B: These are ordinals, not offsets.
    private final int[] _actualOutcomes;
    private final MultivariateOptimizer _optimizer;
    private final int[] _indexList;
    private final S _fromStatus;

    private final double _startingLL;
    private final EntropyCalculator<S, R, T> _calc;

    /**
     * Create a new fitter that is a clone of the old, but using the new params.
     *
     * @param baseFitter_
     * @param params_
     * @param startingLL_
     */
    public CurveParamsFitter(final CurveParamsFitter<S, R, T> baseFitter_, final ItemParameters<S, R, T> params_, final double startingLL_, final EntropyCalculator<S, R, T> calc_)
    {
        if (baseFitter_._fromStatus != params_.getStatus())
        {
            throw new IllegalArgumentException("From status mismatch.");
        }

        //The synchronization doesn't cost much, and improves the correctness.
        synchronized (this)
        {
            synchronized (baseFitter_)
            {
                this._settings = baseFitter_._settings;
                this._factory = baseFitter_._factory;
                this._grid = baseFitter_._grid;
                this._optimizer = baseFitter_._optimizer;
                this._fromStatus = baseFitter_._fromStatus;

                //These can save some resources by copying over.
                this._indexList = baseFitter_._indexList;
                this._actualOutcomes = baseFitter_._actualOutcomes;
            }

            //These we need to build each time...
            final int count = _indexList.length;
            final int reachableCount = _fromStatus.getReachableCount();
            this._model = new ItemModel<>(params_);
            _paramGrid = new ParamFittingGrid<>(_model.getParams(), _grid);
            _powerScores = new RectangularDoubleArray(count, reachableCount);
            fillPowerScores();
            _startingLL = startingLL_;
            _calc = calc_;
        }
    }

    public CurveParamsFitter(final ItemCurveFactory<R, T> factory_,
            final ItemStatusGrid<S, R> grid_, final ItemSettings settings_, final FittingProgressChain<S, R, T> chain_)
    {
        synchronized (this)
        {
            final ItemParameters<S, R, T> params = chain_.getBestParameters();
            _settings = settings_;
            _factory = factory_;
            _model = new ItemModel<>(params);
            _grid = grid_;
            _optimizer = new MultivariateOptimizer(settings_.getBlockSize(), 300, 20, 0.1);
            _fromStatus = params.getStatus();

            final int reachableCount = _fromStatus.getReachableCount();

            _indexList = generateIndexList(_grid, _fromStatus); //Arrays.copyOf(indexList, count);

            final int count = _indexList.length;
            _actualOutcomes = new int[count];

            for (int i = 0; i < count; i++)
            {
                final int index = _indexList[i];
                _actualOutcomes[i] = _grid.getNextStatus(index);
            }

            _paramGrid = new ParamFittingGrid<>(_model.getParams(), _grid);
            _powerScores = new RectangularDoubleArray(count, reachableCount);
            fillPowerScores();
            _startingLL = chain_.getLogLikelihood();
            _calc = chain_.getCalculator();
        }
    }

    public double getEntropy()
    {
        return _startingLL;
    }

    public synchronized RectangularDoubleArray getPowerScores()
    {
        return _powerScores;
    }

    public CurveFitResult<S, R, T> calibrateExistingCurve(final int entryIndex_, final S toStatus_, final double startingLL_) throws ConvergenceException
    {
        final ItemParameters<S, R, T> params = _model.getParams();

        //N.B: Don't check for filtering, this is an entry we already have, filtering isn't relevant.
        final ItemCurveParams<R, T> entryParams = params.getEntryCurveParams(entryIndex_);
        final ItemParameters<S, R, T> reduced = params.dropIndex(entryIndex_);

        CurveFitResult<S, R, T> result = expandParameters(reduced, entryParams, toStatus_, true, startingLL_);
        final double aicDiff = result.calculateAicDifference();

        if (aicDiff > _settings.getAicCutoff())
        {
            //We demand that the AIC improvement is more than the bare minimum. 
            // Also, demand that the resulting diff is statistically significant.
            //LOG.info("AIC improvement is not large enough, keeping old curve.");
            return null;
        }

        return result;
    }

    public CurveFitResult<S, R, T> calibrateCurveAddition(T curveType_, R field_, S toStatus_) throws ConvergenceException
    {
        LOG.info("\nCalculating Curve[" + curveType_ + ", " + field_ + ", " + toStatus_ + "]");

        final QuantileDistribution dist = generateDistribution(field_, toStatus_);
        final ItemCurveParams<R, T> starting = _factory.generateStartingParameters(curveType_, field_, dist, _settings.getRandom());

        final CurveOptimizerFunction<S, R, T> func = generateFunction(starting, toStatus_, false);
        final CurveFitResult<S, R, T> result = generateFit(toStatus_, _model.getParams(), func, _startingLL, starting);

        //final FitResult<S, R, T> output = expandParameters(_model.getParams(), starting, toStatus_, false);
        if (_settings.getPolishStartingParams())
        {
            try
            {
                final ItemCurveParams<R, T> polished = RawCurveCalibrator.polishCurveParameters(_factory, _settings, dist, field_, starting);

                if (polished != starting)
                {
                    //LOG.info("Have polished parameters, testing.");

                    final CurveFitResult<S, R, T> output2 = generateFit(toStatus_, _model.getParams(), func, _startingLL, polished);

                    //LOG.info("Fit comparison: " + output.getLogLikelihood() + " <> " + output2.getLogLikelihood());
                    final double aic1 = result.calculateAicDifference();
                    final double aic2 = output2.calculateAicDifference();
                    final String resString;

                    if (aic1 > aic2)
                    {
                        resString = "BETTER";
                    }
                    else if (aic2 > aic1)
                    {
                        resString = "WORSE";
                    }
                    else
                    {
                        resString = "SAME";
                    }

                    LOG.info("Polished params[" + aic1 + " <> " + aic2 + "]: " + resString);

                    if (aic1 > aic2)
                    {
                        //If the new results are actually better, return those instead.
                        return output2;
                    }
                }
            }
            catch (final Exception e)
            {
                LOG.info("Exception during polish: " + e.toString());
            }
        }

        return result;
    }

    private QuantileDistribution generateDistribution(R field_, S toStatus_)
    {
        final ItemQuantileDistribution<S, R> quantGenerator = new ItemQuantileDistribution<>(_paramGrid, _powerScores, _fromStatus, field_, toStatus_, _indexList);
        final QuantileDistribution dist = quantGenerator.getAdjusted();
        return dist;
    }

    public CurveFitResult<S, R, T> expandParameters(final ItemParameters<S, R, T> params_, final ItemCurveParams<R, T> initParams_, S toStatus_,
            final boolean subtractStarting_) throws ConvergenceException
    {
        final CurveOptimizerFunction<S, R, T> func = generateFunction(initParams_, toStatus_, subtractStarting_);
        final CurveFitResult<S, R, T> result = generateFit(toStatus_, params_, func, _startingLL, initParams_);
        return result;
    }

    public CurveFitResult<S, R, T> expandParameters(final ItemParameters<S, R, T> params_, final ItemCurveParams<R, T> initParams_, S toStatus_,
            final boolean subtractStarting_, final double startingLL_) throws ConvergenceException
    {
        final CurveOptimizerFunction<S, R, T> func = generateFunction(initParams_, toStatus_, subtractStarting_);
        final CurveFitResult<S, R, T> result = generateFit(toStatus_, params_, func, startingLL_, initParams_);
        return result;
    }

    private CurveOptimizerFunction<S, R, T> generateFunction(final ItemCurveParams<R, T> initParams_, S toStatus_, final boolean subtractStarting_)
    {
        final CurveOptimizerFunction<S, R, T> func = new CurveOptimizerFunction<>(initParams_, _factory, _fromStatus, toStatus_, this, _actualOutcomes,
                _paramGrid, _indexList, _settings, subtractStarting_);

        return func;
    }

    public CurveFitResult<S, R, T> generateFit(final S toStatus_, final ItemParameters<S, R, T> baseParams_, final CurveOptimizerFunction<S, R, T> func_,
            final double startingLL_, final ItemCurveParams<R, T> starting_) throws ConvergenceException
    {
        final double[] startingArray = starting_.generatePoint();
        final MultivariatePoint startingPoint = new MultivariatePoint(startingArray);
        final OptimizationResult<MultivariatePoint> result = _optimizer.optimize(func_, startingPoint);

        //Convert the results into params.
        final MultivariatePoint best = result.getOptimum();
        final double[] bestVal = best.getElements();
        final ItemCurveParams<R, T> curveParams = new ItemCurveParams<>(starting_, _factory, bestVal);
        final ItemParameters<S, R, T> updated = baseParams_.addBeta(curveParams, toStatus_);

        //N.B: The optimizer will only run until it is sure that it has found the 
        //best point, or that it can't make further progress. Its goal is not to 
        //compute the log likelihood accurately, so recompute that now. 
        final double recalcEntropy = _calc.computeEntropy(updated).getEntropy();
        final CurveFitResult<S, R, T> output = new CurveFitResult<>(baseParams_, updated, curveParams, toStatus_, recalcEntropy, startingLL_, result.dataElementCount());
        return output;
    }

    public ItemParameters<S, R, T> getParams()
    {
        return _model.getParams();
    }

    private static <S extends ItemStatus<S>, R extends ItemRegressor<R>> int[] generateIndexList(final ItemStatusGrid<S, R> grid_, final S fromStatus_)
    {
        final int gridSize = grid_.size();
        final int fromStatusOrdinal = fromStatus_.ordinal();
        final int[] indexList = new int[gridSize];
        int count = 0;

        for (int i = 0; i < gridSize; i++)
        {
            final int statOrdinal = grid_.getStatus(i);

            if (statOrdinal != fromStatusOrdinal)
            {
                throw new IllegalStateException("Impossible.");
            }
            if (!grid_.hasNextStatus(i))
            {
                throw new IllegalStateException("Impossible.");
            }

            indexList[count++] = i;
        }

        final int[] output = Arrays.copyOf(indexList, count);
        return output;
    }

    private void fillPowerScores()
    {
        final int reachableCount = _fromStatus.getReachableCount();
        final double[] probabilities = new double[reachableCount];
        final int baseCase = _fromStatus.getReachable().indexOf(_fromStatus);
        final int count = _actualOutcomes.length;

        for (int i = 0; i < count; i++)
        {
            final int index = _indexList[i];

            _model.transitionProbability(_paramGrid, index, probabilities);

            MultiLogistic.multiLogitFunction(baseCase, probabilities, probabilities);

            for (int w = 0; w < reachableCount; w++)
            {
                final double next = probabilities[w];
                _powerScores.set(i, w, next);
            }
        }
    }
}
