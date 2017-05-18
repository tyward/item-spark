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
package edu.columbia.tjw.item.algo;

import java.io.Serializable;

/**
 * A two dimensional distribution that allows for easy computation of mean and
 * std. dev. (of the mean) for a 2-D data series.
 *
 *
 * @author tyler
 */
public class DistStats2D implements Serializable
{
    private static final long serialVersionUID = 308684405331906919L;

    private double _xSum;
    private double _x2Sum;
    private double _ySum;
    private double _y2Sum;
    private long _count;

    public DistStats2D()
    {
        this.reset();
    }

    public final double getMeanX()
    {
        return _xSum / _count;
    }

    public final double getMeanY()
    {
        return _ySum / _count;
    }

    /**
     * Actually computes the variance of the mean of X, not the variance of X
     * itself.
     *
     * @return The variance of the mean of X
     */
    public final double getVarX()
    {
        return DistMath.computeMeanVariance(_xSum, _x2Sum, _count);
    }

    public final double getStdDevX()
    {
        return Math.sqrt(getVarX());
    }

    /**
     * Actually computes the variance of the mean of Y, not the variance of Y
     * itself.
     *
     * @return The variance of the mean of Y
     */
    public final double getVarY()
    {
        return DistMath.computeMeanVariance(_ySum, _y2Sum, _count);
    }

    public final double getStdDevY()
    {
        return Math.sqrt(getVarY());
    }

    public final double getXSum()
    {
        return _xSum;
    }

    public final double getX2Sum()
    {
        return _x2Sum;
    }

    public final double getYSum()
    {
        return _ySum;
    }

    public final double getY2Sum()
    {
        return _y2Sum;
    }

    public final long getCount()
    {
        return _count;
    }

    public void update(final double x_, final double y_)
    { 
        final double x2 = x_ * x_;
        final double y2 = y_ * y_;

        _xSum += x_;
        _ySum += y_;
        _x2Sum += x2;
        _y2Sum += y2;
        _count++;
    }

    public final void reset()
    {
        _xSum = 0.0;
        _x2Sum = 0.0;
        _ySum = 0.0;
        _y2Sum = 0.0;
        _count = 0;
    }

}
