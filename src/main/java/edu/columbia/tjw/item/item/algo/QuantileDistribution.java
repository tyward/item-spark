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

import edu.columbia.tjw.item.algo.QuantApprox.QuantileNode;
import edu.columbia.tjw.item.data.InterpolatedCurve;
import edu.columbia.tjw.item.util.MathFunctions;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author tyler
 */
public final class QuantileDistribution implements Serializable
{
    private static final long serialVersionUID = 1116222802982789849L;

    private final double[] _eX;
    private final double[] _devX;
    private final double[] _eY;
    private final double[] _devY;
    private final long[] _count;
    private final long _totalCount;

    //Some stats on the global distribution (across all buckets). 
    private final double _meanX;
    private final double _meanY;
    private final double _meanDevX;
    private final double _meanDevY;

    public QuantileDistribution(final double[] eX_, final double[] eY_, final double[] devX_, final double[] devY_, final long[] count_, final boolean doCopy_)
    {
        final int size = eX_.length;

        if (size != eY_.length || size != devX_.length || size != devY_.length || size != count_.length)
        {
            throw new IllegalArgumentException("Length mismatch.");
        }

        double sumX = 0.0;
        double sumY = 0.0;
        double sumX2 = 0.0;
        double sumY2 = 0.0;

        long count = 0;

        if (doCopy_)
        {
            _eX = eX_.clone();
            _eY = eY_.clone();
            _devX = devX_.clone();
            _devY = devY_.clone();
            _count = count_.clone();
        }
        else
        {
            _eX = eX_;
            _eY = eY_;
            _devX = devX_;
            _devY = devY_;
            _count = count_;
        }

        for (int i = 0; i < size; i++)
        {
            final long bucketCount = _count[i];
            final double eXTerm = _eX[i];
            final double eYTerm = _eY[i];

            final double termX = eXTerm * bucketCount;
            final double termY = eYTerm * bucketCount;
            final double termX2 = bucketCount * eXTerm * eXTerm;
            final double termY2 = bucketCount * eYTerm * eYTerm;
            final double bucketVarX = _devX[i] * _devX[i] * bucketCount;

            sumX += termX;
            sumX2 += termX2;
            sumY += termY;
            sumY2 += termY2;

            count += bucketCount;
        }

        _meanX = sumX / count;
        _meanY = sumY / count;

        _meanDevX = Math.sqrt(DistMath.computeMeanVariance(sumX, sumX2, count));
        _meanDevY = Math.sqrt(DistMath.computeMeanVariance(sumY, sumY2, count));

        _totalCount = count;
    }

    public QuantileDistribution(final QuantApprox approx_)
    {
        final List<QuantileNode> nodes = new ArrayList<>(approx_.size());

        for (final QuantileNode next : approx_)
        {
            if (next.getCount() > 0)
            {
                nodes.add(next);
            }
        }

        final int approxSize = nodes.size();

        _eX = new double[approxSize];
        _devX = new double[approxSize];
        _eY = new double[approxSize];
        _devY = new double[approxSize];
        _count = new long[approxSize];

        long totalCount = 0;

        int pointer = 0;

        for (final QuantileNode next : nodes)
        {
            _eX[pointer] = next.getMeanX();
            _devX[pointer] = next.getStdDevX();
            _eY[pointer] = next.getMeanY();
            _devY[pointer] = next.getStdDevY();

            final long lc = next.getCount();
            _count[pointer] = lc;
            totalCount += lc;
            pointer++;
        }

        for (int i = 0; i < approxSize - 1; i++)
        {
            final double a = _eX[i];
            final double b = _eX[i + 1];

            //Check that they have the same sign. 
            boolean approxEqual = !(a * b <= 0);

            if (approxEqual)
            {
                approxEqual = (MathFunctions.doubleCompareRounded(Math.abs(a), Math.abs(b)) == 0);
            }

            if (!approxEqual)
            {
                continue;
            }

            //Due to rounding, these buckets could actually be in the wrong order.
            if (a > b)
            {
                _eX[i] = b;
                _eX[i + 1] = a;
            }
        }

        _totalCount = totalCount;
        _meanX = approx_.getMeanX();
        _meanY = approx_.getMeanY();
        _meanDevX = approx_.getStdDevX();
        _meanDevY = approx_.getStdDevY();
    }

    public QuantileDistribution alphaTrim(final double alpha_)
    {
        if (alpha_ == 0)
        {
            return this;
        }

        if (alpha_ < 0 || alpha_ >= 0.5)
        {
            throw new IllegalArgumentException("Alpha (for trimming) must be in [0, 0.5]: " + alpha_);
        }

        final int steps = size();
        final int first_step = (int) (alpha_ * steps);
        final int last_step = steps - first_step;
        final int remaining = last_step - first_step;

        if (remaining < 1)
        {
            throw new IllegalArgumentException("Alpha would result in zero steps: " + alpha_);
        }

        if (remaining >= steps)
        {
            return this;
        }

        final double[] eX = new double[remaining];
        final double[] eY = new double[remaining];
        final double[] devX = new double[remaining];
        final double[] devY = new double[remaining];
        final long[] count = new long[remaining];

        for (int i = 0; i < remaining; i++)
        {
            eX[i] = this.getMeanX(first_step + i);
            eY[i] = this.getMeanY(first_step + i);
            devX[i] = this.getDevX(first_step + i);
            devY[i] = this.getDevY(first_step + i);
            count[i] = this.getCount(first_step + i);
        }

        final QuantileDistribution reduced = new QuantileDistribution(eX, eY, devX, devY, count, false);
        return reduced;
    }

    public InterpolatedCurve getDevYCurve(final boolean linear_)
    {
        return new InterpolatedCurve(_eX, _devY, linear_, false);
    }

    public InterpolatedCurve getDevXCurve(final boolean linear_)
    {
        return new InterpolatedCurve(_eX, _devX, linear_, false);
    }

    public InterpolatedCurve getCountCurve(final boolean linear_)
    {
        final int size = this.size();
        final double[] countDoubles = new double[size];

        for (int i = 0; i < size; i++)
        {
            countDoubles[i] = (double) this.getCount(i);
        }

        return new InterpolatedCurve(_eX, countDoubles, linear_, false);
    }

    public InterpolatedCurve getValueCurve(final boolean linear_)
    {
        return new InterpolatedCurve(_eX, _eY, linear_, false);
    }

    public double getMeanX()
    {
        return _meanX;
    }

    public double getMeanY()
    {
        return _meanY;
    }

    public double getDevX()
    {
        return _meanDevX * Math.sqrt(_totalCount);
    }

    public double getDevY()
    {
        return _meanDevY * Math.sqrt(_totalCount);
    }

    public double getMeanDevX()
    {
        return _meanDevX;
    }

    public double getMeanDevY()
    {
        return _meanDevY;
    }

    public long getTotalCount()
    {
        return _totalCount;
    }

    public int size()
    {
        return _count.length;
    }

    public long getCount(final int index_)
    {
        return _count[index_];
    }

    public double getMeanX(final int index_)
    {
        return _eX[index_];
    }

    public double getMeanY(final int index_)
    {
        return _eY[index_];
    }

    public double getDevX(final int index_)
    {
        return _devX[index_];
    }

    public double getDevY(final int index_)
    {
        return _devY[index_];
    }

}
