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

import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.algo.QuantileDistribution;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.util.LogLikelihood;
import edu.columbia.tjw.item.util.MultiLogistic;
import edu.columbia.tjw.item.util.QuantileStatistics;
import edu.columbia.tjw.item.util.RectangularDoubleArray;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author tyler
 * @param <S> The status type for this quantile distribution
 * @param <R> The regressor type for this quantile distribution
 */
public final class ItemQuantileDistribution<S extends ItemStatus<S>, R extends ItemRegressor<R>>
{
    private final LogLikelihood<S> _likelihood;

    private final QuantileDistribution _orig;
    private final QuantileDistribution _adjusted;

    public ItemQuantileDistribution(final ParamFittingGrid<S, R, ?> grid_, final RectangularDoubleArray powerScores_, final S fromStatus_, R field_, S toStatus_, final int[] indexList_)
    {
        this(grid_, powerScores_, fromStatus_, grid_.getRegressorReader(field_), toStatus_, indexList_);
    }

    public ItemQuantileDistribution(final ParamFittingGrid<S, R, ?> grid_, final RectangularDoubleArray powerScores_, final S fromStatus_, final ItemRegressorReader reader_, S toStatus_, final int[] indexList_)
    {
        _likelihood = new LogLikelihood<>(fromStatus_);

        final ItemRegressorReader wrapped = new WrappedRegressorReader(reader_, indexList_);
        final ItemRegressorReader yReader = new InnerResponseReader<>(toStatus_, grid_, powerScores_, _likelihood, indexList_);

        final QuantileStatistics stats = new QuantileStatistics(wrapped, yReader);
        _orig = stats.getDistribution();

        final int size = _orig.size();

        double[] adjY = new double[size];
        double[] eX = new double[size];
        double[] devX = new double[size];
        double[] devAdjY = new double[size];
        long[] bucketCounts = new long[size];

        int pointer = 0;

        for (int i = 0; i < size; i++)
        {
            //We want next / exp(adjustment) = 1.0
            final double next = _orig.getMeanY(i);
            final double nextDev = _orig.getDevY(i);
            final long nextCount = _orig.getCount(i);

            if (nextCount < 1)
            {
                continue;
            }

            //We are actually looking at something like E[actual / predicted]
            final double nextMin = 0.5 / nextCount; //We can't justify a probability smaller than this given our observation count. 
            final double nextMax = 1.0 / nextMin; //1.0 - nextMin;
            final double boundedNext = Math.max(nextMin, Math.min(nextMax, next));

            final double adjustment = Math.log(boundedNext);

            //The operating theory here is that dev is small relative to the adjustment, so we can approximate this...
            final double adjDev = Math.log(nextDev + boundedNext) - adjustment;

            adjY[pointer] = adjustment;
            devAdjY[pointer] = adjDev;

            eX[pointer] = _orig.getMeanX(i);
            devX[pointer] = _orig.getDevX(i);

            bucketCounts[pointer] = _orig.getCount(i);

            pointer++;
        }

        if (pointer < adjY.length)
        {
            adjY = Arrays.copyOf(adjY, pointer);
            devAdjY = Arrays.copyOf(devAdjY, pointer);
            eX = Arrays.copyOf(eX, pointer);
            devX = Arrays.copyOf(devX, pointer);
            bucketCounts = Arrays.copyOf(bucketCounts, pointer);
        }

        _adjusted = new QuantileDistribution(eX, adjY, devX, devAdjY, bucketCounts, false);
    }

    public QuantileDistribution getOrig()
    {
        return _orig;
    }

    public QuantileDistribution getAdjusted()
    {
        return _adjusted;
    }

    private static final class WrappedRegressorReader implements ItemRegressorReader
    {
        private final ItemRegressorReader _underlying;
        private final int[] _indexMap;

        public WrappedRegressorReader(final ItemRegressorReader underlying_, final int[] indexMap_)
        {
            _indexMap = indexMap_;
            _underlying = underlying_;
        }

        @Override
        public double asDouble(int index_)
        {
            final int mapped = _indexMap[index_];
            return _underlying.asDouble(mapped);
        }

        @Override
        public int size()
        {
            return _indexMap.length;
        }

    }

    private static final class InnerResponseReader<S extends ItemStatus<S>, R extends ItemRegressor<R>> implements ItemRegressorReader
    {
        private final double[] _workspace;
        private final int[] _toStatusOrdinals;
        private final RectangularDoubleArray _powerScores;
        private final ParamFittingGrid<S, R, ?> _grid;
        private final LogLikelihood<S> _likelihood;
        private final int[] _indexList;

        public InnerResponseReader(final S toStatus_, final ParamFittingGrid<S, R, ?> grid_, final RectangularDoubleArray powerScores_, final LogLikelihood<S> likelihood_, final int[] indexList_)
        {
            _grid = grid_;
            _powerScores = powerScores_;
            _likelihood = likelihood_;

            _workspace = new double[_powerScores.getColumns()];
            //_toStatusOrdinal = toStatus_.ordinal();

            final List<S> indi = toStatus_.getIndistinguishable();
            _toStatusOrdinals = new int[indi.size()];

            for (int i = 0; i < indi.size(); i++)
            {
                _toStatusOrdinals[i] = indi.get(i).ordinal();
            }

            _indexList = indexList_;
        }

        @Override
        public double asDouble(int index_)
        {
            final int mapped = _indexList[index_];

            if (!_grid.hasNextStatus(mapped))
            {
                throw new IllegalArgumentException("Impossible.");
            }

            for (int k = 0; k < _workspace.length; k++)
            {
                _workspace[k] = _powerScores.get(index_, k);
            }

            final int statusIndex = _grid.getNextStatus(mapped);
            //final int offset = _likelihood.ordinalToOffset(statusIndex);
            MultiLogistic.multiLogisticFunction(_workspace, _workspace);

            double probSum = 0.0;
            double actValue = 0.0;

            for (int i = 0; i < _toStatusOrdinals.length; i++)
            {
                final int nextOffset = _likelihood.ordinalToOffset(_toStatusOrdinals[i]);
                probSum += _workspace[nextOffset];

                if (_toStatusOrdinals[i] == statusIndex)
                {
                    //We took this transition (or one indistinguishable from it). 
                    actValue = 1.0;

                    //Don't break out, we need to sum up all the probability mass...
                    //break;
                }
            }

            //N.B: We know that we will never divide by zero here. 
            //Ideally, this ratio is approximately 1, or at least E[ratio] = 1. 
            //We can compute -ln(1/E[ratio]) and that will give us a power score adjustment we can
            //use to improve our fit. Notice that we are ignoring the non-multiplcative nature of the logistic function. 
            //We will need to run the optimizer over this thing eventually, but this should give us a good starting point. 
            final double ratio = (actValue / probSum);

            //final double residual = (actValue - probSum);
            return ratio;
        }

        @Override
        public int size()
        {
            return _indexList.length;
        }

    }

}
