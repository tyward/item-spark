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

import edu.columbia.tjw.item.base.raw.RawReader;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
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
 */
public class SparkGridAdapter implements ItemStatusGrid<SimpleStatus, SimpleRegressor>
{

    private final SimpleStatus _fromStatus;
    private final EnumFamily<SimpleRegressor> _regFamily;
    private final int[] _toLabels;
    private final ItemRegressorReader[] _readers;

    public SparkGridAdapter(final Dataset<?> data_, final String labelColumn_,
            final String featureColumn_, final List<SimpleRegressor> regressors_, final SimpleStatus fromStatus_, final SimpleRegressor intercept_)
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
            final Number numericLabel = (Number) next.getAs(next.fieldIndex(labelColumn_));
            final String label = Integer.toString(numericLabel.intValue());

            if (vec.size() != regCount)
            {
                throw new IllegalArgumentException("Size mismatch.");
            }

            for (int i = 0; i < regCount; i++)
            {
                transposed[i][pointer] = vec.apply(i);
            }

            final SimpleStatus toStatus = _fromStatus.getFamily().getFromName(label);

            if(null == toStatus) {
                throw new NullPointerException("Unable to find status label: '" + label + "'");
            }

            _toLabels[pointer] = toStatus.ordinal();
            pointer++;
        }

        for (int i = 0; i < regCount; i++)
        {
            final SimpleRegressor reg = regressors_.get(i);
            final RawReader wrapped = new RawReader(transposed[i]);
            _readers[reg.ordinal()] = wrapped;
        }

        _readers[intercept_.ordinal()] = new InterceptReader(rowCount);
    }

    @Override
    public EnumFamily<SimpleStatus> getStatusFamily()
    {
        return _fromStatus.getFamily();
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
    public Set<SimpleRegressor> getAvailableRegressors() {
        return _regFamily.getMembers();
    }

    @Override
    public ItemRegressorReader getRegressorReader(SimpleRegressor field_)
    {
        return _readers[field_.ordinal()];
    }

    @Override
    public int size()
    {
        return _toLabels.length;
    }

    @Override
    public EnumFamily<SimpleRegressor> getRegressorFamily()
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
