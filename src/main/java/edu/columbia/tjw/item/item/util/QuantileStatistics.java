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

import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.algo.QuantApprox;
import edu.columbia.tjw.item.algo.QuantApprox.QuantileNode;
import edu.columbia.tjw.item.algo.QuantileDistribution;

/**
 *
 * @author tyler
 */
public final class QuantileStatistics
{
    private static final int BLOCK_SIZE = 10 * 1000;
    private static final double SIGMA_LIMIT = 3;
    private static final double RELATIVE_ERROR_THRESHOLD = 100.0;

    //private final QuantApprox _approx;
//    private final double[] _eX;
//    private final double[] _eY;
//    private final double[] _varY;
    private final boolean _varTestPassed;
//    private final InterpolatedCurve _curve;
//    private final double _meanX;
//    private final double _meanY;
//    private final double _devX;

    private final QuantileDistribution _dist;

    public QuantileStatistics(final ItemRegressorReader xReader_, final ItemRegressorReader yReader_)
    {
        this(xReader_, yReader_, QuantApprox.DEFAULT_BUCKETS);
    }

    public QuantileStatistics(final ItemRegressorReader xReader_, final ItemRegressorReader yReader_, final int bucketCount_)
    {
        final QuantApprox approx = new QuantApprox(bucketCount_, QuantApprox.DEFAULT_LOAD);

        final int size = xReader_.size();

        if (yReader_.size() != size)
        {
            throw new IllegalArgumentException("Size mismatch.");
        }

        boolean passes = false;
        final double limit2 = SIGMA_LIMIT * SIGMA_LIMIT;

        //We will change these sizes later.
        double[] eX = new double[0];
        double[] eY = eX;
        double[] varY = eX;

        for (int i = 0; i < size; i++)
        {
            final double x = xReader_.asDouble(i);
            final double y = yReader_.asDouble(i);

            approx.addObservation(x, y, true);

            if (0 == (i + 1) % BLOCK_SIZE)
            {
                int index = 0;
                final int approxSize = approx.size();

                if (eX.length != approxSize)
                {
                    eX = new double[approxSize];
                    eY = new double[approxSize];
                    varY = new double[approxSize];
                }

                //let's check out the variance info. 
                for (final QuantileNode next : approx)
                {
                    eX[index] = next.getMeanX();
                    eY[index] = next.getMeanY();
                    varY[index] = next.getVarY();
                    index++;
                }

                passes = true;

                final double globalMeanY = approx.getMeanY();

                for (int w = 1; w < approx.size(); w++)
                {
                    final double ya = eY[w - 1];
                    final double yb = eY[w];
                    final double va = varY[w - 1];
                    final double vb = varY[w];

                    final double diff = (ya - yb);
                    final double d2 = diff * diff;

                    if (Math.abs(ya) > RELATIVE_ERROR_THRESHOLD * Math.sqrt(va))
                    {
                        //If the bucket is very large compared to its relative error, then accept it as-is. 
                        continue;
                    }

                    //Same as saying that the root squared diff is greater than several sigma. 
                    //i.e. we pulled in enough data that we can distinguish between these buckets. 
                    if (d2 < (limit2 * va) || d2 < (limit2 * vb))
                    {
                        final double dMean = (ya - globalMeanY);
                        final double dm2 = dMean * dMean;

                        //They are allowed by either being different from neighboring buckets, or different 
                        //from the mean. Essentially, we don't penalize buckets for being in a flat tail, provided
                        //that tail is not right at the mean (in which case, we really don't know much...). 
                        if (dm2 < (limit2 * va))
                        {
                            passes = false;
                            break;
                        }
                    }
                }

                if (passes)
                {
                    break;
                }
            }
        }

        _dist = approx.getDistribution();

        _varTestPassed = passes;
    }

    public QuantileDistribution getDistribution()
    {
        return _dist;
    }

    public boolean getVarTestPassed()
    {
        return _varTestPassed;
    }
}
