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
public final class UnivariatePoint implements EvaluationPoint<UnivariatePoint>, Comparable<UnivariatePoint>
{
    private static final UnivariatePoint ZERO = new UnivariatePoint(0.0);

    private double _x;

    public UnivariatePoint(final double x_)
    {
        _x = x_;
    }

    public double getValue()
    {
        return _x;
    }

    @Override
    public double getMagnitude()
    {
        return distance(ZERO);
    }

    @Override
    public double distance(UnivariatePoint point_)
    {
        final double x2 = point_.getValue();
        final double diff = Math.abs(_x - x2);
        return diff;
    }

    @Override
    public void scale(double input_)
    {
        _x = _x * input_;
    }

    @Override
    public void add(UnivariatePoint point_)
    {
        final double x2 = point_.getValue();
        _x += x2;
    }

    @Override
    public int compareTo(UnivariatePoint o)
    {
        if (null == o)
        {
            return 1;
        }

        final double x2 = o.getValue();
        final double diff = _x - x2;

        if (diff > 0)
        {
            return 1;
        }
        else if (diff < 0)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    @Override
    public UnivariatePoint clone()
    {
        try
        {
            final UnivariatePoint output = (UnivariatePoint) super.clone();
            return output;
        }
        catch (final CloneNotSupportedException e_)
        {
            throw new RuntimeException(e_);
        }
    }

    @Override
    public void normalize()
    {
        _x = Math.signum(_x);

        if (0.0 == _x)
        {
            _x = 1.0;
        }
    }

    @Override
    public void copy(UnivariatePoint point_)
    {
        this._x = point_.getValue();
    }

    @Override
    public double project(UnivariatePoint input_)
    {
        final double magA = this.getMagnitude();
        final double magB = input_.getMagnitude();
        final double dot = this._x * input_.getValue();

        final double projection = dot / (magB * magA);
        return projection;
    }

}
