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
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.optimize.EvaluationResult;
import edu.columbia.tjw.item.optimize.MultivariateDifferentiableFunction;
import edu.columbia.tjw.item.optimize.MultivariateGradient;
import edu.columbia.tjw.item.optimize.MultivariatePoint;
import edu.columbia.tjw.item.optimize.ThreadedMultivariateFunction;

/**
 *
 * @author tyler
 * @param <S> The status type for this grid
 * @param <R> The regressor type for this grid
 * @param <T> The curve type for this grid
 */
public class LogisticModelFunction<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
        extends ThreadedMultivariateFunction implements MultivariateDifferentiableFunction
{
    private final double[] _beta;
    private final int[] _statusPointers;
    private final int[] _regPointers;
    private final ParamFittingGrid<S, R, T> _grid;
    private ItemParameters<S, R, T> _params;
    private ItemModel<S, R, T> _model;

    public LogisticModelFunction(final double[] beta_, final int[] statusPointers_, final int[] regPointers_,
            final ItemParameters<S, R, T> params_, final ParamFittingGrid<S, R, T> grid_, final ItemModel<S, R, T> model_, ItemSettings settings_)
    {
        super(settings_.getThreadBlockSize(), settings_.getUseThreading());
        _beta = beta_.clone();
        _statusPointers = statusPointers_;
        _regPointers = regPointers_;
        _params = params_;
        _grid = grid_;
        _model = model_;
    }

    public double[] getBeta()
    {
        return _beta.clone();
    }

    public ItemParameters<S, R, T> generateParams(final double[] beta_)
    {
        final ItemParameters<S, R, T> updated = updateParams(_params, _statusPointers, _regPointers, beta_);
        return updated;
    }

    @Override
    public int dimension()
    {
        return _beta.length;
    }

    @Override
    public int numRows()
    {
        return _grid.size();
    }

    @Override
    protected void prepare(MultivariatePoint input_)
    {
        final int dimension = this.dimension();
        boolean changed = false;

        for (int i = 0; i < dimension; i++)
        {
            final double value = input_.getElement(i);

            if (value != _beta[i])
            {
                _beta[i] = value;
                changed = true;
            }
        }

        if (!changed)
        {
            return;
        }

        final ItemParameters<S, R, T> updated = generateParams(_beta);
        _params = updated;
        _model = new ItemModel<>(_params);
    }

    @Override
    protected void evaluate(int start_, int end_, EvaluationResult result_)
    {
        if (start_ == end_)
        {
            return;
        }

        final ItemModel<S, R, T> localModel = _model.clone();
        final S fromStatus = this._model.getParams().getStatus();

        final int fromStatusOrdinal = fromStatus.ordinal();

        for (int i = start_; i < end_; i++)
        {
            final int statOrdinal = _grid.getStatus(i);

            if (statOrdinal != fromStatusOrdinal)
            {
                throw new IllegalStateException("Impossible.");
            }
            if (!_grid.hasNextStatus(i))
            {
                throw new IllegalStateException("Impossible.");
            }

            final double ll = localModel.logLikelihood(_grid, i);

            result_.add(ll, result_.getHighWater(), i + 1);
        }

        result_.setHighRow(end_);
    }

    private ItemParameters<S, R, T> updateParams(final ItemParameters<S, R, T> params_, final int[] rowPointers_, final int[] colPointers_, final double[] betas_)
    {
        final double[][] beta = params_.getBetas();

        for (int i = 0; i < betas_.length; i++)
        {
            final int row = rowPointers_[i];
            final int column = colPointers_[i];
            final double value = betas_[i];
            beta[row][column] = value;
        }

        final ItemParameters<S, R, T> updated = params_.updateBetas(beta);
        return updated;
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

        final ItemModel<S, R, T> localModel = _model.clone();

        final int count = localModel.computeDerivative(_grid, start_, end_, _regPointers, _statusPointers, derivative);

        if (count > 0)
        {
            //N.B: we are computing the negative log likelihood. 
            final double invCount = -1.0 / count;

            for (int i = 0; i < dimension; i++)
            {
                derivative[i] = derivative[i] * invCount;
            }
        }

        final MultivariatePoint der = new MultivariatePoint(derivative);

        final MultivariateGradient grad = new MultivariateGradient(input_, der, null, 0.0);

        return grad;
    }

    @Override
    public int resultSize(int start_, int end_)
    {
        return (end_ - start_);
    }

}
