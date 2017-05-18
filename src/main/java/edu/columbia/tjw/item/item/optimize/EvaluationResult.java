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
package edu.columbia.tjw.item.optimize;

/**
 *
 * @author tyler
 */
public final class EvaluationResult
{
    private final double[] _results;
    private double _sum;
    private double _sqSum;
    private int _highWater;
    private int _resetCount;
    private int _highRow;

    public EvaluationResult(final int size_)
    {
        _results = new double[size_];
        _resetCount = 0;
        this.clear();
    }

    public double getSum()
    {
        return _sum;
    }

    public double getMean()
    {
        if (0 == _highWater)
        {
            return 0.0;
        }

        return _sum / _highWater;
    }

    /**
     * Return the variance of the mean.
     *
     * @return The variance of the mean
     */
    public double getVariance()
    {
        if (0 == _highWater)
        {
            return 0.0;
        }

        final double invN = 1.0 / _highWater;
        final double eX = _sum * invN;
        final double eX2 = _sqSum * invN;
        final double variance = eX2 - (eX * eX);
        final double meanVariance = invN * variance;

        if (meanVariance < 0.0)
        {
            //Protect against rounding issues here. 
            return 0.0;
        }

        return meanVariance;
    }

    public double getStdDev()
    {
        final double variance = getVariance();
        final double stdDev = Math.sqrt(variance);
        return stdDev;
    }

    public int getHighWater()
    {
        return _highWater;
    }

    public int getHighRow()
    {
        return _highRow;
    }

    public void setHighRow(final int endRow_)
    {
        if (endRow_ < _highRow)
        {
            throw new IllegalArgumentException("High water must only increase.");
        }

        _highRow = endRow_;
    }

    public void add(final double observation_, final int startValue_, final int endRow_)
    {
        setHighRow(endRow_);
        _sum += observation_;
        _sqSum += (observation_ * observation_);

        if (_highWater != startValue_)
        {
            throw new IllegalArgumentException("Must add data in order!");
        }

        _results[startValue_] = observation_;
        _highWater++;

    }

    public int getResetCount()
    {
        return _resetCount;
    }

    public double get(final int row_)
    {
        if (row_ >= _highWater)
        {
            throw new IllegalArgumentException("Data not yet set.");
        }

        return _results[row_];
    }

    public void add(final EvaluationResult result_, final int startValue_, final int endRow_)
    {
        setHighRow(endRow_);

        if (_highWater != startValue_)
        {
            throw new IllegalArgumentException("Must add data in order!");
        }

        final int addCount = result_.getHighWater();

        if (addCount + _highWater > this._results.length)
        {
            System.out.println("Boing.");
        }

        if (addCount > 0)
        {
            System.arraycopy(result_._results, 0, this._results, this._highWater, addCount);
            _sum += result_.getSum();
            _sqSum += result_._sqSum;
        }

        _highWater += addCount;
    }

    public void clear()
    {
        _sum = 0.0;
        _sqSum = 0.0;
        _highWater = 0;
        _highRow = 0;
        _resetCount++;
    }
}
