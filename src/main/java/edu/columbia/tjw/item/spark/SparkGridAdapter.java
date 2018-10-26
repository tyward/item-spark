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
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.base.RawReader;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.util.EnumFamily;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 */
public class SparkGridAdapter<S extends ItemStatus<S>, R extends ItemRegressor<R>> implements ItemStatusGrid<S, R>
{

    private final S _fromStatus;
    private final EnumFamily<R> _regFamily;
    private final int[] _toLabels;
    private final ItemRegressorReader[] _readers;

    public SparkGridAdapter(final Dataset<?> data_, final String labelColumn_,
            final String featureColumn_, final List<R> regressors_, final S fromStatus_, final R intercept_)
    {
        _fromStatus = fromStatus_;
        final int rowCount = (int) data_.count();
        final int regCount = regressors_.size();

        _regFamily = intercept_.getFamily();
        _readers = new ItemRegressorReader[_regFamily.size()];

        final Iterator<Row> rowForm = (Iterator<Row>) data_.toLocalIterator();

        final double[][] transposed = new double[regCount][rowCount];
        _toLabels = new int[rowCount];
        int pointer = 0;

        while (rowForm.hasNext())
        {
            final Row next = rowForm.next();
            final Vector vec = (Vector) next.get(next.fieldIndex(featureColumn_));
            final Number label = (Number) next.getAs(next.fieldIndex(labelColumn_));

            if (vec.size() != regCount)
            {
                throw new IllegalArgumentException("Size mismatch.");
            }

            for (int i = 0; i < regCount; i++)
            {
                transposed[i][pointer] = vec.apply(i);
            }

            _toLabels[pointer] = label.intValue();
            pointer++;
        }

        for (int i = 0; i < regCount; i++)
        {
            final R reg = regressors_.get(i);
            final RawReader wrapped = new RawReader(transposed[i]);
            _readers[reg.ordinal()] = wrapped;
        }

        _readers[intercept_.ordinal()] = new InterceptReader(rowCount);
    }

    @Override
    public EnumFamily<S> getStatusFamily()
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getStatus(int index_)
    {
        return _fromStatus.ordinal();
    }

    @Override
    public int getNextStatus(int index_)
    {
        return _toLabels[index_];
    }

    @Override
    public boolean hasNextStatus(int index_)
    {
        return true;
    }

    @Override
    public Set<R> getAvailableRegressors() {
        return _regFamily.getMembers();
    }

    @Override
    public ItemRegressorReader getRegressorReader(R field_)
    {
        return _readers[field_.ordinal()];
    }

    @Override
    public int size()
    {
        return _toLabels.length;
    }

    @Override
    public EnumFamily<R> getRegressorFamily()
    {
        return _regFamily;
    }

    private static final class InterceptReader implements ItemRegressorReader
    {

        private final int _size;

        public InterceptReader(final int size_)
        {
            _size = size_;
        }

        @Override
        public double asDouble(int index_)
        {
            if (index_ < 0 || index_ >= _size)
            {
                throw new ArrayIndexOutOfBoundsException("Index out of bounds[0, " + _size + "): " + index_);
            }

            return 1.0;
        }

        @Override
        public int size()
        {
            return _size;
        }

    }


}
