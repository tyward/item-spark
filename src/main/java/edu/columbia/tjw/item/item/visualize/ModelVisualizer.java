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
package edu.columbia.tjw.item.visualize;

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.algo.QuantApprox;
import edu.columbia.tjw.item.algo.QuantileDistribution;
import edu.columbia.tjw.item.data.InterpolatedCurve;
import edu.columbia.tjw.item.data.ItemGrid;
import edu.columbia.tjw.item.fit.ItemCalcGrid;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.util.EnumFamily;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 *
 * @author tyler
 * @param <S> The status family for this visualizer
 * @param <R> The regressor family for this visualizer
 * @param <T> The curve family for this visualizer
 */
public class ModelVisualizer<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements Serializable
{
    private static final long serialVersionUID = 4198554232525136207L;
    private final ItemParameters<S, R, T> _params;
    private final SortedSet<S> _reachable;
    private final SortedSet<R> _regressors;
    private final SortedMap<S, SortedMap<R, QuantileDistribution>> _distMap;
    private final SortedMap<S, SortedMap<R, QuantileDistribution>> _modelMap;

    public ModelVisualizer(final ItemParameters<S, R, T> params_, final ParamFittingGrid<S, R, T> grid_, final SortedSet<R> extraRegressors_)
    {
        this(params_, grid_, extraRegressors_, QuantApprox.DEFAULT_BUCKETS, QuantApprox.DEFAULT_LOAD);
    }

    public ModelVisualizer(final ItemParameters<S, R, T> params_, final ParamFittingGrid<S, R, T> grid_, final SortedSet<R> extraRegressors_, final int approxBuckets_, final int approxLoad_)
    {
        if (approxBuckets_ < 10)
        {
            throw new IllegalArgumentException("Bucket count must be at least 10: " + approxBuckets_);
        }

        _params = params_;
        final ItemModel<S, R, T> model = new ItemModel<>(params_);
        final ParamFittingGrid<S, R, T> grid = grid_;
        final S from = params_.getStatus();

        final TreeSet<R> basic = new TreeSet<>(params_.getUniqueRegressors());
        basic.addAll(extraRegressors_);

        _regressors = Collections.unmodifiableSortedSet(basic);

        if (from.getReachableCount() < 2)
        {
            //Terminal state, skip.
            throw new IllegalArgumentException("Cannot visualize a terminal state.");
        }

        _reachable = Collections.unmodifiableSortedSet(new TreeSet<>(from.getReachable()));
        final int fromOrdinal = from.ordinal();
        final SortedMap<S, SortedMap<R, QuantileDistribution>> distMap = new TreeMap<>();
        final SortedMap<S, SortedMap<R, QuantileDistribution>> modelMap = new TreeMap<>();

        final double[] workspace = new double[_reachable.size()];

        for (final S to : _reachable)
        {
            final SortedMap<R, QuantileDistribution> approximations = new TreeMap<>();
            final SortedMap<R, QuantileDistribution> modelCalcs = new TreeMap<>();
            final int toIndex = from.getReachable().indexOf(to);

            for (final R reg : _regressors)
            {
                final QuantApprox approx = new QuantApprox(approxBuckets_, approxLoad_);
                final QuantApprox modelApprox = new QuantApprox(approxBuckets_, approxLoad_);

                final ItemRegressorReader reader = grid.getRegressorReader(reg);

                final int toOrdinal = to.ordinal();

                for (int i = 0; i < grid.size(); i++)
                {
                    if (fromOrdinal != grid.getStatus(i))
                    {
                        continue;
                    }
                    if (!grid.hasNextStatus(i))
                    {
                        continue;
                    }

                    final int trueToOrdinal = grid.getNextStatus(i);
                    final double prob;

                    if (trueToOrdinal == toOrdinal)
                    {
                        prob = 1.0;
                    }
                    else
                    {
                        prob = 0.0;
                    }

                    final double x = reader.asDouble(i);
                    approx.addObservation(x, prob, true);

                    model.transitionProbability(grid, i, workspace);
                    final double modelProb = workspace[toIndex];
                    modelApprox.addObservation(x, modelProb);
                }

                approximations.put(reg, new QuantileDistribution(approx));
                modelCalcs.put(reg, new QuantileDistribution(modelApprox));
            }

            distMap.put(to, Collections.unmodifiableSortedMap(approximations));
            modelMap.put(to, Collections.unmodifiableSortedMap(modelCalcs));
        }

        _distMap = Collections.unmodifiableSortedMap(distMap);
        _modelMap = Collections.unmodifiableSortedMap(modelMap);
    }

    public SortedSet<R> getRegressors()
    {
        return _regressors;
    }

    public S getFrom()
    {
        return _params.getStatus();
    }

    public SortedSet<S> getReachable()
    {
        return _reachable;
    }

    private static <S extends ItemStatus<S>, R extends ItemRegressor<R>>
            QuantileDistribution extractDistribution(final S to_, final R reg_, SortedMap<S, SortedMap<R, QuantileDistribution>> map_)
    {
        if (!map_.containsKey(to_))
        {
            throw new IllegalArgumentException("Invalid to state.");
        }

        final SortedMap<R, QuantileDistribution> regMap = map_.get(to_);

        if (!regMap.containsKey(reg_))
        {
            throw new IllegalArgumentException("Invalid to regressor.");
        }

        return regMap.get(reg_);
    }

    public InterpolatedCurve generateQQPlot(final S to_, final R reg_)
    {
        final InterpolatedCurve modelCurve = this.graph(to_, reg_, CurveType.MODEL, 0.01);
        final InterpolatedCurve actualCurve = this.graph(to_, reg_, CurveType.ACTUAL, 0.01);

        final int size = modelCurve.size();

        final double[] y = new double[size];

        double eY2 = 0.0;

        for (int i = 0; i < modelCurve.size(); i++)
        {
            final double currX = modelCurve.getX(i);
            final double modY = modelCurve.getY(i);
            final double actY = actualCurve.value(currX);
            final double residual = actY - modY;
            y[i] = residual;

            eY2 += (residual * residual);
        }

        eY2 /= size;
        Arrays.sort(y);

        final double devY = Math.sqrt(eY2);

        final double[] x = new double[size];
        final NormalDistribution dist = new NormalDistribution(0, devY);
        final double normLength = size * 1.02;

        for (int i = 0; i < y.length; i++)
        {
            final double currX = (i + 0.01) / normLength;
            final double expected = dist.inverseCumulativeProbability(currX);
            x[i] = expected;
        }

        final InterpolatedCurve output = new InterpolatedCurve(x, y);
        return output;
    }

    public QuantileDistribution getModelDistribution(final S to_, final R reg_)
    {
        return extractDistribution(to_, reg_, _modelMap);
    }

    public QuantileDistribution getActualDistribution(final S to_, final R reg_)
    {
        return extractDistribution(to_, reg_, _distMap);
    }

    public InterpolatedCurve graph(final S to_, final R regressor_, final CurveType type_)
    {
        return graph(to_, regressor_, type_, 0.05);
    }

    public InterpolatedCurve graph(final S to_, final R regressor_, final CurveType type_, final double alpha_)
    {
        switch (type_)
        {
            case THEORETICAL:
            {
                final Map<R, Double> regValues = new TreeMap<>();

                for (final R reg : this.getRegressors())
                {
                    final QuantileDistribution regDist = getActualDistribution(to_, reg);

                    //Maybe update to an alpha trimmed mean as well?
                    final double mean = regDist.getMeanX();
                    regValues.put(reg, mean);
                }

                final QuantileDistribution dist = getActualDistribution(to_, regressor_);
                final QuantileDistribution reduced = dist.alphaTrim(alpha_);

                final double regMin = reduced.getMeanX(0);
                final double regMax = reduced.getMeanX(reduced.size() - 1);

                final InterpolatedCurve curve = graph(to_, regressor_, regValues, regMin, regMax, reduced.size());
                return curve;
            }
            case ACTUAL:
            {
                final QuantileDistribution dist = getActualDistribution(to_, regressor_);
                final QuantileDistribution reduced = dist.alphaTrim(alpha_);
                return reduced.getValueCurve(true);
            }
            case MODEL:
            {
                final QuantileDistribution dist = getModelDistribution(to_, regressor_);
                final QuantileDistribution reduced = dist.alphaTrim(alpha_);
                return reduced.getValueCurve(true);
            }
            case MASS:
            {
                final QuantileDistribution dist = getActualDistribution(to_, regressor_);
                final QuantileDistribution reduced = dist.alphaTrim(alpha_);
                return reduced.getCountCurve(true);
            }
            default:
                throw new IllegalArgumentException("Unknown Curve Type.");
        }
    }

    /**
     * Holding all other regressors fixed, we compute the probability of the
     * given transition for various values of this target regressor.
     *
     * The result is two arrays, the regressor values and transition
     * probabilities.
     *
     * @param to_ The status transition to target
     * @param regressor_ The regressor to target
     * @param regValues_ Values to use for all regressors (except possibly
     * regressor_)
     * @param regMin_ The starting value for regressor_
     * @param regMax_ The ending value for regressor_
     * @param steps_ The number of steps to use
     * @return An interpolated curve over regressor_ on [regMin_, regMax_] of
     * transition probabilities for an element with regValues_
     */
    public InterpolatedCurve graph(final S to_, final R regressor_, final Map<R, Double> regValues_, final double regMin_, final double regMax_, final int steps_)
    {
//        if (regMin_ >= regMax_)
//        {
//            throw new IllegalArgumentException("Reg min must be greater than reg max.");
//        }
//        if (steps_ <= 0)
//        {
//            throw new IllegalArgumentException("Steps must be positive: " + steps_);
//        }

        final double stepSize = (regMax_ - regMin_) / steps_;

        final ItemModel<S, R, T> model = new ItemModel<>(_params);
        final InnerGrid grid = new InnerGrid(steps_, regMin_, stepSize, regressor_, regValues_);
        final ItemCalcGrid<S, R, T> paramGrid = new ItemCalcGrid<>(_params, grid);

        final List<S> reachable = _params.getStatus().getReachable();
        final int toIndex = reachable.indexOf(to_);

        if (toIndex < 0)
        {
            throw new IllegalArgumentException("Status not reachable: " + to_);
        }

        final double[] probability = new double[reachable.size()];

        final double[] x = new double[steps_];
        final double[] y = new double[steps_];

        for (int i = 0; i < steps_; i++)
        {
            model.transitionProbability(paramGrid, i, probability);

            x[i] = regMin_ + (i * stepSize);
            y[i] = probability[toIndex];
        }

        final InterpolatedCurve output = new InterpolatedCurve(x, y, true, false);
        return output;
    }

    private final class InnerGrid implements ItemGrid<R>
    {
        private final int _steps;
        private final R _regressor;
        private final ItemRegressorReader[] _readers;

        public InnerGrid(final int steps_, final double minValue_, final double stepSize_, final R regressor_, final Map<R, Double> regValues_)
        {
            _steps = steps_;
            _regressor = regressor_;

            _readers = new ItemRegressorReader[regressor_.getFamily().size()];

            for (final Map.Entry<R, Double> entry : regValues_.entrySet())
            {
                final R next = entry.getKey();
                final Double value = entry.getValue();
                _readers[next.ordinal()] = new ConstantRegressorReader(_steps, value);
            }

            _readers[regressor_.ordinal()] = new SteppedRegressorReader(_steps, minValue_, stepSize_);
        }

        @Override
        public int size()
        {
            return _steps;
        }

        @Override
        public ItemRegressorReader getRegressorReader(R field_)
        {
            return _readers[field_.ordinal()];
        }

        @Override
        public EnumFamily<R> getRegressorFamily()
        {
            return _regressor.getFamily();
        }

        @Override
        public boolean hasRegressorReader(R field_)
        {
            return _readers[field_.ordinal()] != null;
        }

    }

    private static final class ConstantRegressorReader implements ItemRegressorReader
    {
        private final int _size;
        private final double _regValue;

        public ConstantRegressorReader(final int size_, final double regValue_)
        {
            _regValue = regValue_;
            _size = size_;

        }

        @Override
        public double asDouble(int index_)
        {
            return _regValue;
        }

        @Override
        public int size()
        {
            return _size;
        }

    }

    private static final class SteppedRegressorReader implements ItemRegressorReader
    {
        private final int _size;
        private final double _startValue;
        private final double _stepValue;

        public SteppedRegressorReader(final int size_, final double startValue_, final double stepValue_)
        {
            _startValue = startValue_;
            _stepValue = stepValue_;
            _size = size_;
        }

        @Override
        public double asDouble(int index_)
        {
            final double reg = _startValue + (index_ * _stepValue);
            return reg;
        }

        @Override
        public int size()
        {
            return _size;
        }

    }

    public enum CurveType
    {
        THEORETICAL,
        ACTUAL,
        MODEL,
        MASS
    }

}
