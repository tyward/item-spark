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

import edu.columbia.tjw.item.util.HashUtil;
import java.util.Arrays;

/**
 *
 * @author tyler
 */
public class MultivariatePoint implements EvaluationPoint<MultivariatePoint>
{
    private double[] _point;

    public MultivariatePoint(final double[] raw_)
    {
        _point = raw_.clone();

        for (int i = 0; i < _point.length; i++)
        {
            if (Double.isNaN(_point[i]) || Double.isInfinite(_point[i]))
            {
                throw new IllegalArgumentException("Points must be well defined: " + this.toString());
            }
        }
    }

    public MultivariatePoint(final MultivariatePoint copyFrom_)
    {
        this(copyFrom_.getDimension());
        this.copy(copyFrom_);
    }

    public MultivariatePoint(final int size_)
    {
        _point = new double[size_];
    }

    public double[] getElements()
    {
        final double[] output = _point.clone();
        return output;
    }

    public void setElements(final double[] data_)
    {
        final int size = _point.length;

        if (data_.length != size)
        {
            throw new IllegalArgumentException("Incorrect length: " + data_.length + " != " + size);
        }

        for (int i = 0; i < size; i++)
        {
            setElement(i, data_[i]);
        }
    }

    public void setElement(final int index_, final double value_)
    {
        if (Double.isNaN(value_) || Double.isInfinite(value_))
        {
            throw new IllegalArgumentException("Points must be well defined.");
        }

        _point[index_] = value_;
    }

    public double getElement(final int index_)
    {
        return _point[index_];
    }

    @Override
    public double project(MultivariatePoint input_)
    {
        if (this.getDimension() != input_.getDimension())
        {
            throw new IllegalArgumentException("Dimensionality must match.");
        }

        final double thatMagnitude = input_.getMagnitude();

        final int dimension = this.getDimension();
        double sum = 0.0;

        for (int i = 0; i < dimension; i++)
        {
            final double a = this.getElement(i);
            final double b = input_.getElement(i);
            sum += (a * b);
        }

        final double output = sum / (thatMagnitude);
        return output;
    }

    @Override
    public double getMagnitude()
    {
        double sum = 0.0;

        for (int i = 0; i < _point.length; i++)
        {
            sum += _point[i] * _point[i];
        }

        return Math.sqrt(sum);
    }

    @Override
    public double distance(MultivariatePoint point_)
    {
        double sum = 0.0;

        for (int i = 0; i < _point.length; i++)
        {
            final double term = (_point[i] - point_._point[i]);

            sum += term * term;
        }

        return Math.sqrt(sum);

    }

    @Override
    public void scale(double input_)
    {
        if (Double.isNaN(input_) || Double.isInfinite(input_))
        {
            throw new IllegalArgumentException("Points must be well defined: " + input_);
        }

        for (int i = 0; i < _point.length; i++)
        {
            _point[i] = input_ * _point[i];
        }
    }

    @Override
    public void copy(MultivariatePoint point_)
    {
        checkLength(point_);
        System.arraycopy(point_._point, 0, _point, 0, _point.length);
    }

    @Override
    public void add(MultivariatePoint point_)
    {
        checkLength(point_);

        for (int i = 0; i < _point.length; i++)
        {
            _point[i] += point_._point[i];
        }
    }

    @Override
    public void normalize()
    {
        final double mag = this.getMagnitude();

        if (Double.isNaN(mag) || Double.isInfinite(mag))
        {
            throw new IllegalArgumentException("Points must be well defined: " + this.toString());
        }

        if (mag == 0.0)
        {
            throw new IllegalStateException("Cannot normalize the zero point.");
        }

        final double invMag = 1.0 / mag;

        for (int i = 0; i < _point.length; i++)
        {
            _point[i] = _point[i] * invMag;
        }
    }

    @Override
    public MultivariatePoint clone()
    {
        try
        {
            final MultivariatePoint point = (MultivariatePoint) super.clone();
            point._point = point._point.clone();
            return point;

        }
        catch (final CloneNotSupportedException e)
        {
            throw new RuntimeException(e);
        }
    }

    public int getDimension()
    {
        return _point.length;
    }

    private void checkLength(final MultivariatePoint point_)
    {
        if (point_._point.length != _point.length)
        {
            throw new IllegalArgumentException("Length mismatch.");
        }
    }

    @Override
    public int hashCode()
    {
        int hash = HashUtil.startHash(MultivariatePoint.class);

        for (int i = 0; i < this._point.length; i++)
        {
            hash = HashUtil.mix(hash, Double.doubleToLongBits(this._point[i]));
        }

        return hash;
    }

    @Override
    public boolean equals(final Object that_)
    {
        if (null == that_)
        {
            return false;
        }
        if (this == that_)
        {
            return true;
        }
        if (that_.getClass() != this.getClass())
        {
            return false;
        }

        final boolean result = equals((MultivariatePoint) that_);
        return result;
    }

    public boolean equals(final MultivariatePoint that_)
    {
        if (null == that_)
        {
            return false;
        }
        if (this == that_)
        {
            return true;
        }

        final boolean equal = Arrays.equals(this._point, that_._point);
        return equal;
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();
        builder.append(this.getClass().getName() + ": " + Arrays.toString(_point));
        return builder.toString();
    }

}
