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
package edu.columbia.tjw.item.util;

/**
 *
 * @author tyler
 */
public class RectangularDoubleArray
{
    private final double[] _data;
    private final int _rows;
    private final int _columns;

    public RectangularDoubleArray(final int rows_, final int columns_)
    {
        final int size = rows_ * columns_;

        _rows = rows_;
        _columns = columns_;
        _data = new double[size];
    }

    public double get(final int index_)
    {
        final double output = _data[index_];
        return output;
    }

    public void set(final int index_, final double value_)
    {
        _data[index_] = value_;
    }

    public double get(final int row_, final int column_)
    {
        final int index = computeIndex(row_, column_);
        return get(index);
    }

    public void set(final int row_, final int column_, final double value_)
    {
        final int index = computeIndex(row_, column_);
        set(index, value_);
    }

    public int getRows()
    {
        return _rows;
    }

    public int getColumns()
    {
        return _columns;
    }

    public int size()
    {
        return _data.length;
    }

    /**
     * Check for errors that will not cause an array index out of bounds
     * exception.
     *
     * @param row_
     * @param column_
     */
    private int computeIndex(final int row_, final int column_)
    {
        if (column_ < 0)
        {
            throw new ArrayIndexOutOfBoundsException("Column must be non-negative: " + column_);
        }
        if (column_ >= _columns)
        {
            throw new ArrayIndexOutOfBoundsException("Column too large: " + column_);
        }
        if (row_ > _rows)
        {
            throw new ArrayIndexOutOfBoundsException("Row too large: " + row_);
        }

        final int index = (row_ * _columns) + column_;
        return index;
    }

}
