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
import edu.columbia.tjw.item.data.ItemGrid;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.util.EnumFamily;
import java.util.List;

/**
 *
 * @author tyler
 * @param <S> The status type for this param grid
 * @param <R> The regressor type for this param grid
 * @param <T> The curve type for this param grid
 */
public abstract class ItemParamGrid<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements ItemGrid<R>
{
    private final ItemParameters<S, R, T> _params;
    private final List<R> _uniqueRegressors;
    private final ItemRegressorReader[] _readers;
    private final int _uniqueCount;

    public ItemParamGrid(final ItemParameters<S, R, T> params_, final ItemGrid<R> grid_)
    {
        _params = params_;
        _uniqueRegressors = _params.getUniqueRegressors();

        _uniqueCount = _uniqueRegressors.size();
        _readers = new ItemRegressorReader[_uniqueCount];

        for (int i = 0; i < _uniqueCount; i++)
        {
            final R next = _uniqueRegressors.get(i);
            _readers[i] = grid_.getRegressorReader(next);
        }
    }

    public abstract ItemGrid<R> getUnderlying();

    /**
     * A short hand function. First creates an array of the correct size, then
     * calls getRegessors(index_, array)
     *
     * @param index_
     * @return
     */
    public double[] getRegressors(final int index_)
    {
        final double[] regVector = new double[_readers.length];
        getRegressors(index_, regVector);
        return regVector;
    }

    /**
     * Get the regressors for this observation.
     *
     * Note that the results will be transformed (according to the curves from
     * getTransformation)
     *
     * @param index_ The row index of the regressors
     * @param output_ The array of regressors, ordered as per the transformation
     * set.
     */
    public void getRegressors(final int index_, final double[] output_)
    {
        if (output_.length != _readers.length)
        {
            throw new IllegalArgumentException("Size mismatch: " + output_.length + " != " + _readers.length);
        }

        for (int i = 0; i < _readers.length; i++)
        {
            output_[i] = _readers[i].asDouble(index_);
        }
    }

    @Override
    public ItemRegressorReader getRegressorReader(R field_)
    {
        return this.getUnderlying().getRegressorReader(field_);
    }

    @Override
    public int size()
    {
        return this.getUnderlying().size();
    }

    @Override
    public EnumFamily<R> getRegressorFamily()
    {
        return this.getUnderlying().getRegressorFamily();
    }

    @Override
    public boolean hasRegressorReader(R field_)
    {
        return this.getUnderlying().hasRegressorReader(field_);
    }

}
