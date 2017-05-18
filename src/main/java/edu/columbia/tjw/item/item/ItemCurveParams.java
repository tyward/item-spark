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
package edu.columbia.tjw.item;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author tyler
 * @param <R>
 * @param <T> The curve type defined by these params
 */
public final class ItemCurveParams<R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements Serializable
{
    private static final long serialVersionUID = 0x7b981aa6c028bfa7L;

    private static final int INTERCEPT_INDEX = 0;
    private static final int BETA_INDEX = 1;
    private static final int NON_CURVE_PARAM_COUNT = 2;

    private final int _size;
    private final double _intercept;
    private final double _beta;
    private final List<T> _types;
    private final List<R> _regressors;
    private final List<ItemCurve<T>> _curves;

    //These are used to track which parameter of which underlying curve a given 
    // element of the curvePoint_ array points to. This is important for 
    //calculating derivatives.
    private final int[] _curveOffsets;
    private final int[] _curveIndices;

    public ItemCurveParams(final T type_, final R field_, ItemCurveFactory<R, T> factory_, final double[] curvePoint_)
    {
        this(Collections.singletonList(type_), Collections.singletonList(field_), factory_, curvePoint_[INTERCEPT_INDEX], curvePoint_[BETA_INDEX], NON_CURVE_PARAM_COUNT, curvePoint_);
    }

    public ItemCurveParams(final List<T> types_, final List<R> fields_, ItemCurveFactory<R, T> factory_, final double intercept_, final double beta_, final int arrayOffset_, final double[] curvePoint_)
    {
        if (Double.isNaN(intercept_))
        {
            throw new IllegalArgumentException("Intercept must be well defined.");
        }
        if (Double.isNaN(beta_))
        {
            throw new IllegalArgumentException("Beta must be well defined.");
        }
        if (types_.size() != fields_.size())
        {
            throw new IllegalArgumentException("Size mismatch.");
        }

        _types = Collections.unmodifiableList(new ArrayList<>(types_));
        _regressors = Collections.unmodifiableList(new ArrayList<>(fields_));
        _size = calculateSize(_types);

        _intercept = intercept_;
        _beta = beta_;

        _curveOffsets = new int[_size];
        _curveIndices = new int[_size];
        _curveOffsets[INTERCEPT_INDEX] = -1;
        _curveOffsets[BETA_INDEX] = -1;

        _curveIndices[INTERCEPT_INDEX] = -1;
        _curveIndices[BETA_INDEX] = -1;

        int pointer = arrayOffset_;

        List<ItemCurve<T>> curveList = new ArrayList<>(_types.size());

        for (final T next : _types)
        {
            if (null == next)
            {
                curveList.add(null);
                continue;
            }

            ItemCurve<T> curve = factory_.generateCurve(next, pointer, curvePoint_);
            final int paramCount = next.getParamCount();

            //Whatever our array offset, we want this to be the "raw" pointer, 
            // what it would have been if the offset was 2, indicating it starts after intercept and beta...
            final int basePointer = NON_CURVE_PARAM_COUNT + pointer - arrayOffset_;

            for (int i = 0; i < paramCount; i++)
            {
                _curveIndices[basePointer + i] = curveList.size();
                _curveOffsets[basePointer + i] = i;
            }

            pointer += next.getParamCount();
            curveList.add(curve);
        }

        _curves = Collections.unmodifiableList(curveList);
    }

    public ItemCurveParams(final double intercept_, final double beta_, final R field_, final ItemCurve<T> curve_)
    {
        this(intercept_, beta_, Collections.singletonList(field_), Collections.singletonList(curve_));
    }

    public ItemCurveParams(final double intercept_, final double beta_, final List<R> fields_, final List<ItemCurve<T>> curves_)
    {
        if (Double.isNaN(intercept_))
        {
            throw new IllegalArgumentException("Intercept must be well defined.");
        }
        if (Double.isNaN(beta_))
        {
            throw new IllegalArgumentException("Beta must be well defined.");
        }
        if (fields_.size() != curves_.size())
        {
            throw new IllegalArgumentException("List size mismatch: " + fields_.size() + " !+ " + curves_.size());
        }

        final List<T> types = new ArrayList<>(curves_.size());

        for (final ItemCurve<T> curve : curves_)
        {
            if (null == curve)
            {
                types.add(null);
            }
            else
            {
                types.add(curve.getCurveType());
            }
        }

        _types = Collections.unmodifiableList(types);
        _regressors = Collections.unmodifiableList(new ArrayList<>(fields_));
        _curves = Collections.unmodifiableList(new ArrayList<>(curves_));

        _intercept = intercept_;
        _beta = beta_;
        _size = calculateSize(_types);

        _curveOffsets = new int[_size];
        _curveIndices = new int[_size];
        _curveOffsets[INTERCEPT_INDEX] = -1;
        _curveOffsets[BETA_INDEX] = -1;

        _curveIndices[INTERCEPT_INDEX] = -1;
        _curveIndices[BETA_INDEX] = -1;

        int pointer = NON_CURVE_PARAM_COUNT;

        for (int i = 0; i < _types.size(); i++)
        {
            final T next = _types.get(i);

            if (null == next)
            {
                continue;
            }

            final int paramCount = next.getParamCount();

            for (int w = 0; w < paramCount; w++)
            {
                _curveOffsets[pointer + w] = w;
                _curveIndices[pointer + w] = i;
            }
        }

    }

    public ItemCurveParams(final T type_, final R field_, ItemCurveFactory<R, T> factory_, final double intercept_, final double beta_, final double[] curveParams_)
    {
        this(Collections.singletonList(type_), Collections.singletonList(field_), factory_, intercept_, beta_, 0, curveParams_);
    }

    public ItemCurveParams(final ItemCurveParams<R, T> baseParams_, ItemCurveFactory<R, T> factory_, final double[] values_)
    {
        _size = baseParams_.size();
        _types = baseParams_._types;
        _regressors = baseParams_._regressors;
        _intercept = values_[INTERCEPT_INDEX];
        _beta = values_[BETA_INDEX];
        _curveIndices = baseParams_._curveIndices;
        _curveOffsets = baseParams_._curveOffsets;

        int pointer = NON_CURVE_PARAM_COUNT;

        List<ItemCurve<T>> curveList = new ArrayList<>(_types.size());

        for (final T next : _types)
        {
            if (null == next)
            {
                curveList.add(null);
                continue;
            }

            ItemCurve<T> curve = factory_.generateCurve(next, pointer, values_);
            pointer += next.getParamCount();
            curveList.add(curve);
        }

        _curves = Collections.unmodifiableList(curveList);
    }

    private static <T extends ItemCurveType<T>> int calculateSize(final List<T> types_)
    {
        int size = NON_CURVE_PARAM_COUNT;

        for (final T next : types_)
        {
            if (null == next)
            {
                continue;
            }

            size += next.getParamCount();
        }

        return size;
    }

    public int getEffectiveParamCount()
    {
        //Always one less than the size, because we aren't charged for the intercept.
        return (_size - 1);
    }

    public List<T> getTypes()
    {
        return _types;
    }

    public List<R> getRegressors()
    {
        return _regressors;
    }

    public List<ItemCurve<T>> getCurves()
    {
        return _curves;
    }

    public T getType(final int depth_)
    {
        return _types.get(depth_);
    }

    public R getRegressor(final int depth_)
    {
        return _regressors.get(depth_);
    }

    public ItemCurve<T> getCurve(final int depth_)
    {
        return _curves.get(depth_);
    }

    public double getIntercept()
    {
        return _intercept;
    }

    public double getBeta()
    {
        return _beta;
    }

    /**
     * Similar to the notion of entry depth in ItemParameters.
     *
     * @return
     */
    public int getEntryDepth()
    {
        return _types.size();
    }

    /**
     * How many parameters are we dealing with here.
     *
     * Including all curves, bet and intercept.
     *
     * @return
     */
    public int size()
    {
        return _size;
    }

    public int indexToCurveIndex(final int index_)
    {
        final int curveIndex = _curveIndices[index_];
        return curveIndex;
    }

    public int indexToCurveOffset(final int index_)
    {
        final int curveOffset = _curveOffsets[index_];
        return curveOffset;
    }

    public int getInterceptIndex()
    {
        return INTERCEPT_INDEX;
    }

    public int getBetaIndex()
    {
        return BETA_INDEX;
    }

    public void extractPoint(final double[] point_)
    {
        if (point_.length != _size)
        {
            throw new IllegalArgumentException("Invalid point array size.");
        }

        point_[INTERCEPT_INDEX] = _intercept;
        point_[BETA_INDEX] = _beta;

        int pointer = NON_CURVE_PARAM_COUNT;

        for (final ItemCurve<T> next : _curves)
        {
            if (null == next)
            {
                continue;
            }

            for (int i = 0; i < next.getCurveType().getParamCount(); i++)
            {
                point_[pointer++] = next.getParam(i);
            }
        }
    }

    public double[] generatePoint()
    {
        final double[] output = new double[this.size()];
        extractPoint(output);
        return output;
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();
        builder.append("ItemCurveParams[");
        builder.append(this._intercept);
        builder.append(", ");
        builder.append(this._beta);
        builder.append("]:\n");

        for (int i = 0; i < this.getEntryDepth(); i++)
        {
            final ItemCurve<T> curve = this.getCurve(i);
            final R reg = this.getRegressor(i);

            builder.append("\n\t[");
            builder.append(i);
            builder.append("][");
            builder.append(reg);
            builder.append("]: ");
            builder.append(curve);
        }

        final String output = builder.toString();
        return output;
    }

}
