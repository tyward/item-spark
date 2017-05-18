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
public class ResultComparator
{
    private final EvaluationResult _resA;
    private final EvaluationResult _resB;
    private int _rcA;
    private int _rcB;
    private int _compareLimit;
    private double _sum;
    private double _squareSum;

    public ResultComparator(final EvaluationResult resA_, final EvaluationResult resB_)
    {
        _resA = resA_;
        _resB = resB_;

        _rcA = resA_.getResetCount();
        _rcB = resB_.getResetCount();

        _compareLimit = 0;
        _sum = 0.0;
        _squareSum = 0.0;
    }

    public double computeZScore()
    {
        if ((_resA.getResetCount() != _rcA) || (_resB.getResetCount() != _rcB))
        {
            reset();
        }

        final int maxLimit = Math.min(_resA.getHighWater(), _resB.getHighWater());

        for (int i = _compareLimit; i < maxLimit; i++)
        {
            final double a = _resA.get(i);
            final double b = _resB.get(i);

            final double diff = (b - a);
            _sum += diff;
            _squareSum += (diff * diff);
        }

        _compareLimit = maxLimit;

        if (_compareLimit == 0)
        {
            return 0.0;
        }

        final double invN = 1.0 / _compareLimit;
        final double eX = _sum * invN;
        final double eX2 = _squareSum * invN;

        final double varX = eX2 - (eX * eX);
        final double varMu = varX * invN;

        final double sigmaMu;

        if (varMu <= 0.0)
        {
            if (eX == 0.0)
            {
                return 0.0;
            }

            sigmaMu = 0.0;
        }
        else
        {
            sigmaMu = Math.sqrt(varMu);
        }

        final double zScore = eX / sigmaMu;
        return zScore;
    }

    private void reset()
    {
        _rcA = _resA.getResetCount();
        _rcB = _resB.getResetCount();

        _compareLimit = 0;
        _sum = 0.0;
        _squareSum = 0.0;
    }

}
