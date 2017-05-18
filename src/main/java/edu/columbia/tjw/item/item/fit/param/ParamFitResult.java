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
package edu.columbia.tjw.item.fit.param;

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.util.MathFunctions;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public final class ParamFitResult<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private final double _startingLogL;
    private final double _logL;
    private final double _llImprovement;
    private final int _rowCount;
    private final ItemParameters<S, R, T> _starting;
    private final ItemParameters<S, R, T> _endingParams;

    public ParamFitResult(final ItemParameters<S, R, T> starting_, final ItemParameters<S, R, T> ending_, final double logLikelihood_, final double startingLL_, final int rowCount_)
    {
        if (null == starting_ || null == ending_)
        {
            throw new NullPointerException("Parameters cannot be null.");
        }
        if (Double.isNaN(logLikelihood_) || Double.isInfinite(logLikelihood_) || logLikelihood_ < 0.0)
        {
            throw new IllegalArgumentException("Log likelihood must be well defined.");
        }
        if (Double.isNaN(startingLL_) || Double.isInfinite(startingLL_) || startingLL_ < 0.0)
        {
            throw new IllegalArgumentException("Starting Log Likelihood must be well defined.");
        }

        _starting = starting_;
        _endingParams = ending_;
        _startingLogL = startingLL_;

        if (isUnchanged())
        {
            //Don't let strange rounding errors throw us off.
            _logL = _startingLogL;
            _llImprovement = 0.0;
        }
        else
        {
            _logL = logLikelihood_;
            _llImprovement = (startingLL_ - _logL);
        }

        _rowCount = rowCount_;
    }

    public boolean isWorse()
    {
        if (_starting.getEffectiveParamCount() != _endingParams.getEffectiveParamCount())
        {

        }

        return MathFunctions.isAicWorse(_startingLogL, _logL);
    }

    public boolean isBetter()
    {
        return MathFunctions.isAicWorse(_logL, _startingLogL);
    }

    public boolean isUnchanged()
    {
        return (_starting == _endingParams);
    }

    public double getAic()
    {
        final double scaledImprovement = _llImprovement * _rowCount;
        final double paramContribution = (_endingParams.getEffectiveParamCount() - _starting.getEffectiveParamCount());
        final double aicDiff = 2.0 * (paramContribution - scaledImprovement);
        return aicDiff;
    }

    public double getStartingLL()
    {
        return _startingLogL;
    }

    public double getEndingLL()
    {
        return _logL;
    }

    public double getLLImprovement()
    {
        return _llImprovement;
    }

    public ItemParameters<S, R, T> getStartingParams()
    {
        return _starting;
    }

    public ItemParameters<S, R, T> getEndingParams()
    {
        return _endingParams;
    }

}
