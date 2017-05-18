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
package edu.columbia.tjw.item.data;

import java.io.Serializable;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;

/**
 *
 * @author tyler
 */
public final class InterpolatedCurve implements Serializable, UnivariateFunction
{
    private static final long serialVersionUID = 1203431614753798217L;

    private final double[] _x;
    private final double[] _y;
    private final double _minX;
    private final double _maxX;
    private final double _firstY;
    private final double _lastY;
    private final boolean _isLinear;

    private transient UnivariateFunction _spline = null;

    public InterpolatedCurve(final double[] x_, final double[] y_)
    {
        this(x_, y_, false, true);
    }

    public InterpolatedCurve(final double[] x_, final double[] y_, final boolean isLinear_, final boolean doCopy_)
    {
        if (x_.length != y_.length)
        {
            throw new IllegalArgumentException("Length mismatch: " + x_.length + " != " + y_.length);
        }

        for (int i = 0; i < x_.length; i++)
        {
            if (i > 0 && !(x_[i] > x_[i - 1]))
            {
                throw new IllegalArgumentException("X values must be sorted and well defined.");
            }

            if (Double.isInfinite(x_[i]))
            {
                throw new IllegalArgumentException("X values must be finite defined.");
            }
        }

        if (doCopy_)
        {
            _x = x_.clone();
            _y = y_.clone();
        }
        else
        {
            _x = x_;
            _y = y_;
        }

        final int lastIndex = _x.length - 1;

        _minX = _x[0];
        _maxX = _x[lastIndex];

        _firstY = _y[0];
        _lastY = _y[lastIndex];
        _isLinear = isLinear_;
    }

    public int size()
    {
        return _x.length;
    }

    public double getX(final int index_)
    {
        return _x[index_];
    }
    
    public double getY(final int index_)
    {
        return _y[index_];
    }
    
    @Override
    public double value(double x_)
    {

        if (Double.isNaN(x_))
        {
            return Double.NaN;
        }
        if (x_ <= _minX)
        {
            return _firstY;
        }
        if (x_ >= _maxX)
        {
            return _lastY;
        }

        final UnivariateFunction spline = generateInterp();
        final double output = spline.value(x_);
        return output;
    }

    private UnivariateFunction generateInterp()
    {
        if (null != _spline)
        {
            return _spline;
        }

        final UnivariateInterpolator interp;

        if (_isLinear)
        {
            interp = new LinearInterpolator();
        }
        else
        {
            interp = new SplineInterpolator();
        }

        _spline = interp.interpolate(_x, _y);

        return _spline;
    }

}
