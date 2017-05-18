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
public class MultivariateGradient
{
    private final MultivariatePoint _gradient;
    private final MultivariatePoint _secondDerivative;
    private final MultivariatePoint _center;
    private final double _stdDev;

    public MultivariateGradient(final MultivariatePoint center_, final MultivariatePoint gradient_, final MultivariatePoint secondDerivative_, final double stdDev_)
    {
        _gradient = new MultivariatePoint(gradient_);
        _center = new MultivariatePoint(center_);

        if (null == secondDerivative_)
        {
            _secondDerivative = null;
        }
        else
        {
            _secondDerivative = new MultivariatePoint(secondDerivative_);
        }

        _stdDev = stdDev_;
    }

    public MultivariatePoint getSecondDerivative()
    {
        return _secondDerivative;
    }

    public MultivariatePoint getGradient()
    {
        return _gradient;
    }

    public MultivariatePoint getCenter()
    {
        return _center;
    }

    public double getStdDev()
    {
        return _stdDev;
    }
}
