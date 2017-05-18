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
package edu.columbia.tjw.item.fit.param;

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ParamFilter;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.fit.EntropyCalculator;
import edu.columbia.tjw.item.fit.FittingProgressChain;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.optimize.MultivariateOptimizer;
import edu.columbia.tjw.item.optimize.MultivariatePoint;
import edu.columbia.tjw.item.optimize.OptimizationResult;
import edu.columbia.tjw.item.util.LogUtil;
import java.util.Arrays;
import java.util.Collection;
import java.util.logging.Logger;

/**
 *
 * @author tyler
 * @param <S> The status type for this fitter
 * @param <R> The regressor type for this fitter
 * @param <T> The curve type for this fitter
 */
public final class ParamFitter<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private static final Logger LOG = LogUtil.getLogger(ParamFitter.class);

    private final MultivariateOptimizer _optimizer;
    private final ItemSettings _settings;
    private final Collection<ParamFilter<S, R, T>> _filters;
    private final EntropyCalculator<S, R, T> _calc;

    ItemParameters<S, R, T> _cacheParams;
    LogisticModelFunction<S, R, T> _cacheFunction;

    public ParamFitter(final EntropyCalculator<S, R, T> calc_, final ItemSettings settings_, final Collection<ParamFilter<S, R, T>> filters_)
    {
        _calc = calc_;
        _filters = filters_;
        _optimizer = new MultivariateOptimizer(settings_.getBlockSize(), 300, 20, 0.1);
        _settings = settings_;
    }

    public ParamFitResult<S, R, T> fit(final FittingProgressChain<S, R, T> chain_) throws ConvergenceException
    {
        return fit(chain_, chain_.getBestParameters());
    }

    public synchronized ParamFitResult<S, R, T> fit(final FittingProgressChain<S, R, T> chain_, ItemParameters<S, R, T> params_) throws ConvergenceException
    {
        final double entropy = chain_.getLogLikelihood();

        if (params_ != _cacheParams)
        {
            _cacheParams = params_;
            _cacheFunction = generateFunction(params_);
        }

        final LogisticModelFunction<S, R, T> function = _cacheFunction;
        final double[] beta = function.getBeta();
        final MultivariatePoint point = new MultivariatePoint(beta);
        final int numRows = function.numRows();

        final OptimizationResult<MultivariatePoint> result = _optimizer.optimize(function, point);
        final MultivariatePoint optimumPoint = result.getOptimum();

        for (int i = 0; i < beta.length; i++)
        {
            beta[i] = optimumPoint.getElement(i);
        }

        final double newLL = result.minValue();
        LOG.info("Fitting coefficients, LL improvement: " + entropy + " -> " + newLL + "(" + (newLL - entropy) + ")");

        if (!result.converged())
        {
            LOG.info("Exhausted dataset before convergence, moving on.");
        }

        final ParamFitResult<S, R, T> output;

        if (newLL > entropy)
        {
            output = new ParamFitResult<>(chain_.getBestParameters(), chain_.getBestParameters(), entropy, entropy, numRows);
            chain_.pushResults("ParamFit", output.getEndingParams(), output.getEndingLL());
        }
        else
        {
            final ItemParameters<S, R, T> updated = function.generateParams(beta);

            final double recalcEntropy = _calc.computeEntropy(updated).getEntropy();
            output = new ParamFitResult<>(params_, updated, recalcEntropy, entropy, numRows);
            chain_.pushResults("ParamFit", output.getEndingParams(), output.getEndingLL());
        }

        return output;
    }

    private LogisticModelFunction<S, R, T> generateFunction(final ItemParameters<S, R, T> params_)
    {
        final int reachableCount = params_.getStatus().getReachableCount();
        final int entryCount = params_.getEntryCount();

        final S from = params_.getStatus();

        final int maxSize = reachableCount * entryCount;

        int pointer = 0;
        double[] beta = new double[maxSize];
        int[] statusPointers = new int[maxSize];
        int[] regPointers = new int[maxSize];

        for (int i = 0; i < reachableCount; i++)
        {
            final S to = from.getReachable().get(i);

            for (int k = 0; k < entryCount; k++)
            {
                if (params_.betaIsFrozen(to, k, _filters))
                {
                    continue;
                }

                beta[pointer] = params_.getBeta(i, k);
                statusPointers[pointer] = i;
                regPointers[pointer] = k;
                pointer++;
            }
        }

        beta = Arrays.copyOf(beta, pointer);
        statusPointers = Arrays.copyOf(statusPointers, pointer);
        regPointers = Arrays.copyOf(regPointers, pointer);

        final ParamFittingGrid<S, R, T> grid = new ParamFittingGrid<>(params_, _calc.getGrid());
        final LogisticModelFunction<S, R, T> function = new LogisticModelFunction<>(beta, statusPointers, regPointers, params_, grid, new ItemModel<>(params_), _settings);
        return function;
    }

}
