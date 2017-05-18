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

import edu.columbia.tjw.item.util.LogUtil;
import java.util.logging.Logger;

/**
 *
 * @author tyler
 * @param <V> The type of points on which this can be evaluated
 * @param <F> The type of optimization function which will be called
 */
public class BasicAdaptiveComparator<V extends EvaluationPoint<V>, F extends OptimizationFunction<V>> implements AdaptiveComparator<V, F>
{
    private static final Logger LOG = LogUtil.getLogger(BasicAdaptiveComparator.class);
    private final int _blockSize;
    private final double _stdDevThreshold;

    private EvaluationResult _a;
    private EvaluationResult _b;
    private ResultComparator _comp;

    public BasicAdaptiveComparator(final int blockSize_, final double stdDevThreshold_)
    {
        _blockSize = blockSize_;
        _stdDevThreshold = stdDevThreshold_;

        _a = null;
        _b = null;
        _comp = null;
    }

    @Override
    public double compare(final F function_, final V a_, final V b_, final EvaluationResult aResult_, final EvaluationResult bResult_)
    {
        if (a_.distance(b_) == 0.0)
        {
            return 0.0;
        }
        if (aResult_ == bResult_)
        {
            throw new IllegalArgumentException("Results for distinct points must be distinct.");
        }
        if ((_a != aResult_) || (_b != bResult_))
        {
            _comp = new ResultComparator(aResult_, bResult_);
            _a = aResult_;
            _b = bResult_;
        }

        int aCount = aResult_.getHighRow();
        int bCount = bResult_.getHighRow();
        final int numRows = function_.numRows();

        if (aCount < _blockSize)
        {
            final int end = Math.min(numRows, aCount + _blockSize);
            function_.value(a_, aCount, end, aResult_);
            aCount = aResult_.getHighRow();
        }
        if (bCount < _blockSize)
        {
            final int end = Math.min(numRows, bCount + _blockSize);
            function_.value(b_, bCount, end, bResult_);
            bCount = bResult_.getHighRow();
        }

        double zScore = _comp.computeZScore();

        while ((Math.abs(zScore) < _stdDevThreshold) && ((aCount < numRows) || (bCount < numRows)))
        {
            //We don't know enough to tell for sure which one is greater, try to improve our estimate.
            if (aCount < bCount)
            {
                final int end = Math.min(aCount + _blockSize, numRows);
                function_.value(a_, aCount, end, aResult_);
                aCount = aResult_.getHighRow();
            }
            else
            {
                final int end = Math.min(bCount + _blockSize, numRows);
                function_.value(b_, bCount, end, bResult_);
                bCount = bResult_.getHighRow();
            }

            zScore = _comp.computeZScore();
        }

        final double output = -zScore;
        //LOG.info("Point comparison complete: [" + aCount + ", " + bCount + "]: " + output);

        if (Double.isNaN(output) || Double.isInfinite(output))
        {
            LOG.warning("Bad comparison.");
        }

        return output;
    }

    @Override
    public double getSigmaTarget()
    {
        return this._stdDevThreshold;
    }

}
