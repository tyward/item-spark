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
public class GeneralOptimizationResult<V extends EvaluationPoint<V>> implements OptimizationResult<V>
{
    private final V _optimum;
    private final EvaluationResult _minResult;
    private final boolean _converged;
    private final int _evalCount;

    public GeneralOptimizationResult(final V optimum_, final EvaluationResult minResult_, final boolean converged_, final int evalCount_)
    {
        if (null == optimum_)
        {
            throw new NullPointerException("Optimum cannot be null.");
        }
        if (null == minResult_)
        {
            throw new NullPointerException("Evaluation result cannot be null.");
        }
        if (evalCount_ < 0)
        {
            throw new IllegalArgumentException("Eval count must be nonnegative: " + evalCount_);
        }

        _minResult = minResult_;
        _converged = converged_;
        _evalCount = evalCount_;
        _optimum = optimum_;
    }

    @Override
    public final V getOptimum()
    {
        return _optimum;
    }

    @Override
    public final boolean converged()
    {
        return _converged;
    }

    @Override
    public final int evaluationCount()
    {
        return _evalCount;
    }

    @Override
    public final double minValue()
    {
        return _minResult.getMean();
    }

    @Override
    public final EvaluationResult minResult()
    {
        return _minResult;
    }

    @Override
    public int dataElementCount()
    {
        return _minResult.getHighWater();
    }
}
