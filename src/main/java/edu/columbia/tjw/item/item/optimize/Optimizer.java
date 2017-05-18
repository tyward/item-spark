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
 * @param <V> The type of points over which this can optimize
 * @param <F> The type of function this can optimize
 */
public abstract class Optimizer<V extends EvaluationPoint<V>, F extends OptimizationFunction<V>>
{
    private static final double DEFAULT_XTOL = 1.0e-6;
    private static final double DEFAULT_YTOL = 1.0e-6;

    private final double _stdDevThreshold = 5.0;
    private final double _xTol;
    private final double _yTol;
    private final int _blockSize;
    private final int _maxEvalCount;

    private final AdaptiveComparator<V, F> _comparator;

    public Optimizer(final int blockSize_, final int maxEvalCount_)
    {
        this(DEFAULT_XTOL, DEFAULT_YTOL, blockSize_, maxEvalCount_);
    }

    public abstract OptimizationResult<V> optimize(final F f_, final V startingPoint_, final V direction_) throws ConvergenceException;

    public Optimizer(final double xTol_, final double yTol_, final int blockSize_, final int maxEvalCount_)
    {
        _blockSize = blockSize_;
        _xTol = xTol_;
        _yTol = yTol_;
        _maxEvalCount = maxEvalCount_;

        _comparator = new BasicAdaptiveComparator<>(_blockSize, _stdDevThreshold);
    }

    public double getXTolerance()
    {
        return _xTol;
    }

    public double getYTolerance()
    {
        return _yTol;
    }

    public int getBlockSize()
    {
        return _blockSize;
    }

    public int getMaxEvalCount()
    {
        return _maxEvalCount;
    }

    public AdaptiveComparator<V, F> getComparator()
    {
        return _comparator;
    }

    /**
     * Figure out how far apart these two results could realistically be,
     * without attempting additional calculations.
     *
     * @param aResult_ The first result
     * @param bResult_ The second result
     * @return True if aResult_.getMean() is within tolerance of
     * bResult_.getMean()
     */
    protected boolean checkYTolerance(final EvaluationResult aResult_, final EvaluationResult bResult_)
    {
        final double meanA = aResult_.getMean();
        final double meanB = bResult_.getMean();

        final double stdDevA = aResult_.getStdDev();
        final double stdDevB = bResult_.getStdDev();

        final double raw = Math.abs(meanA - meanB);
        final double adjusted = raw + this._stdDevThreshold * (stdDevA + stdDevB);

        final double scale = Math.abs((meanA * meanA) + (meanB * meanB));

        final double scaled = adjusted / scale;

        final boolean output = scaled < this._yTol;
        return output;
    }

    protected boolean checkYTolerance(final EvaluationResult aResult_, final EvaluationResult bResult_, final EvaluationResult cResult_)
    {
        final boolean checkA = checkYTolerance(aResult_, bResult_);
        final boolean checkB = checkYTolerance(bResult_, cResult_);

        final boolean output = checkA && checkB;
        return output;
    }

    protected boolean checkXTolerance(final V a_, final V b_)
    {
        final double distance = a_.distance(b_);

        final double aMag = a_.getMagnitude();
        final double bMag = b_.getMagnitude();

        final double scale = Math.sqrt((aMag * aMag) + (bMag * bMag));

        final double result = distance / scale;

        final boolean output = result < this._xTol;
        return output;
    }

}
