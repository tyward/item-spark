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
package edu.columbia.tjw.item.fit;

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.data.RandomizedStatusGrid;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public final class EntropyCalculator<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private final ItemStatusGrid<S, R> _grid;
    private final S _fromStatus;

    public EntropyCalculator(final ItemStatusGrid<S, R> grid_, final S fromStatus_, final ItemSettings settings_)
    {
        _fromStatus = fromStatus_;

        if (grid_ instanceof RandomizedStatusGrid)
        {
            _grid = grid_;
        }
        else
        {
            final ItemStatusGrid<S, R> wrapped = new RandomizedStatusGrid<>(grid_, settings_, grid_.getRegressorFamily(), fromStatus_);
            _grid = wrapped;
        }

    }

    public ItemStatusGrid<S, R> getGrid()
    {
        return _grid;
    }

    public S getFromStatus()
    {
        return _fromStatus;
    }

    public int size()
    {
        return _grid.size();
    }

    public EntropyAnalysis computeEntropy(final ItemParameters<S, R, T> params_)
    {
        if (params_.getStatus() != getFromStatus())
        {
            throw new IllegalArgumentException("Status mismatch.");
        }

        final ParamFittingGrid<S, R, T> grid = new ParamFittingGrid<>(params_, _grid);
        final ItemModel<S, R, T> model = new ItemModel<>(params_);

        final S fromStatus = params_.getStatus();
        final int fromStatusOrdinal = fromStatus.ordinal();

        int count = 0;
        double entropySum = 0.0;
        double x2 = 0.0;

        for (int i = 0; i < grid.size(); i++)
        {
            final int statOrdinal = _grid.getStatus(i);

            if (statOrdinal != fromStatusOrdinal)
            {
                continue;
            }
            if (!_grid.hasNextStatus(i))
            {
                continue;
            }

            final double entropy = model.logLikelihood(grid, i);
            final double e2 = entropy * entropy;
            entropySum += entropy;
            count++;
            x2 += e2;
        }

        if (count <= 0)
        {
            return new EntropyAnalysis(0.0, 0.0, 0);
        }

        final double eX = entropySum / count;
        final double eX2 = x2 / count;
        final double varianceX = Math.max(0.0, eX2 - (eX * eX));
        final double varianceMu = varianceX / count;
        final double sigma = Math.sqrt(varianceMu);

        return new EntropyAnalysis(eX, sigma, count);
    }

    public static final class EntropyAnalysis
    {
        private final double _entropy;
        private final double _sigma;
        private final int _size;

        public EntropyAnalysis(final double entropy_, final double sigma_, final int size_)
        {
            _entropy = entropy_;
            _sigma = sigma_;
            _size = size_;
        }

        public double getEntropy()
        {
            return _entropy;
        }

        public double getSigma()
        {
            return _sigma;
        }

        public int getSize()
        {
            return _size;
        }

    }

}
