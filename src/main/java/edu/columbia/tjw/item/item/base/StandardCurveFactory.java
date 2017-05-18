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
package edu.columbia.tjw.item.base;

import edu.columbia.tjw.item.ItemCurve;
import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveParams;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.algo.QuantileDistribution;
import edu.columbia.tjw.item.util.EnumFamily;
import edu.columbia.tjw.item.util.MathFunctions;
import java.util.Random;
import org.apache.commons.math3.util.FastMath;

/**
 * This is the default implementation of the curve factory.
 *
 * This provides curves for the standard curve types described in the paper
 *
 * Unless there is good reason, use this as the curve factory.
 *
 *
 * @author tyler
 * @param <R>
 */
public final class StandardCurveFactory<R extends ItemRegressor<R>> implements ItemCurveFactory<R, StandardCurveType>
{
    private static final double SLOPE_MULT = 10.0;
    private static final long serialVersionUID = 0xfa6df5b97c865a49L;

    public StandardCurveFactory()
    {
    }

    @Override
    public ItemCurve<StandardCurveType> generateCurve(StandardCurveType type_, int offset_, double[] params_)
    {
        final double centralityVal = params_[offset_];
        final double slopeParam = params_[offset_ + 1];

        switch (type_)
        {
            case LOGISTIC:
                return new LogisticCurve(centralityVal, slopeParam);
            case GAUSSIAN:
                return new GaussianCurve(centralityVal, slopeParam);
            default:
                throw new RuntimeException("Impossible, unknown type: " + type_);
        }
    }

    @Override
    public ItemCurveParams<R, StandardCurveType> generateStartingParameters(final StandardCurveType type_, final R field_, final QuantileDistribution dist_, final Random rand_)
    {
        final double[] curveParams = new double[2]; //== type_.getParamCount();

        //First, choose the mean uniformly....
        final int size = dist_.size();
        final double meanSelector = rand_.nextDouble();
        final double invCount = 1.0 / dist_.getTotalCount();
        double runningSum = 0.0;
        double xVal = 0.0;
        int xIndex = 0;

        for (int i = 0; i < size; i++)
        {
            final long bucketCount = dist_.getCount(i);
            final double frac = bucketCount * invCount;
            xVal = dist_.getMeanX(i);
            xIndex = i;

            runningSum += frac;

            if (runningSum > meanSelector)
            {
                break;
            }
        }

        curveParams[0] = xVal;

        final double distDev = dist_.getDevX();

        //This is a reasonable estimate of how low the std. dev can realistically be. 
        //Given that we have seen only finitely many observations, we could never say with confidence that it's zero.
        //Also, in case the mean happens to be zero, and the dev is zero, then we add a tiny value to not divide by zero.
        final double minDev = 1.0e-10 + Math.abs(dist_.getMeanX() / dist_.getTotalCount());

        final double slopeParam;
        final double betaGuess;

        switch (type_)
        {
            case LOGISTIC:
                // The logic is that we'll want one unit of slope to occupy something 
                // like the midpoint between 1 bucket and all the buckets, so about sqrt(buckets)
                // However, randomize this a bit to give us more chances to get it right. 
                final double slopeScale = (0.5 + rand_.nextDouble()) * Math.sqrt(size);

                //The square root is because the slope is squared before being applied, to keep the logistic upward sloping.
                slopeParam = Math.sqrt(slopeScale / Math.max(minDev, distDev));
                //slopeParam = Math.sqrt(1.0 / (distDev + 1.0e-10));

                double xCorrelation = 0.0;
                //double xVar = 0.0;
                final double meanY = dist_.getMeanY();
                final double meanX = dist_.getMeanX();

                for (int i = 0; i < size; i++)
                {
                    final double yDev = dist_.getMeanY(i) - meanY;
                    final double xDev = dist_.getMeanX(i) - meanX;
                    final double corr = yDev * xDev * dist_.getCount(i);
                    xCorrelation += corr;
                }

                final double xDev = dist_.getDevX();
                final double yDev = dist_.getDevY();
                final double obsCount = dist_.getTotalCount();

                //Between -1.0 and 1.0, a reasonable guess for beta...
                betaGuess = xCorrelation / (obsCount * xDev * yDev);

                break;
            case GAUSSIAN:
                slopeParam = (0.5 + rand_.nextDouble()) * Math.max(minDev, distDev);
                betaGuess = dist_.getMeanY(xIndex) - dist_.getMeanY();
                break;
            default:
                throw new RuntimeException("Impossible.");
        }

        curveParams[1] = slopeParam;

        final double intercept = -0.5 * betaGuess;

        final ItemCurveParams<R, StandardCurveType> output = new ItemCurveParams<>(type_, field_, this, intercept, betaGuess, curveParams);
        return output;

    }

    @Override
    public EnumFamily<StandardCurveType> getFamily()
    {
        return StandardCurveType.FAMILY;
    }

    @Override
    public ItemCurve<StandardCurveType> boundCentrality(ItemCurve<StandardCurveType> inputCurve_, double lowerBound_, double upperBound_)
    {
        if (null == inputCurve_)
        {
            return null;
        }

        final StandardCurveType type = inputCurve_.getCurveType();

        switch (type)
        {
            case LOGISTIC:
            {
                final LogisticCurve curve = (LogisticCurve) inputCurve_;

                final double center = curve._center;

                if (center < lowerBound_)
                {
                    return new LogisticCurve(lowerBound_, Math.sqrt(curve._slope));
                }
                if (center > upperBound_)
                {
                    return new LogisticCurve(upperBound_, Math.sqrt(curve._slope));
                }

                return curve;
            }
            case GAUSSIAN:
            {
                final GaussianCurve curve = (GaussianCurve) inputCurve_;
                final double center = curve._mean;

                if (center < lowerBound_)
                {
                    return new GaussianCurve(lowerBound_, Math.sqrt(curve._stdDev));
                }
                if (center > upperBound_)
                {
                    return new GaussianCurve(upperBound_, Math.sqrt(curve._stdDev));
                }

                return curve;
            }
            default:
                throw new RuntimeException("Impossible.");
        }

    }

    private static final class GaussianCurve extends StandardCurve<StandardCurveType>
    {
        private static final long serialVersionUID = 0xd1c81f26497f177fL;
        private final double _stdDev;
        private final double _invStdDev;
        private final double _mean;
        private final double _expNormalizer;

        public GaussianCurve(final double mean_, final double stdDev_)
        {
            super(StandardCurveType.GAUSSIAN);

            if (Double.isInfinite(mean_) || Double.isNaN(mean_))
            {
                throw new IllegalArgumentException("Invalid mean: " + mean_);
            }
            if (Double.isInfinite(stdDev_) || Double.isNaN(stdDev_))
            {
                throw new IllegalArgumentException("Invalid stdDev: " + stdDev_);
            }

            final double variance = (stdDev_ * stdDev_) + 1.0e-10;
            _stdDev = Math.sqrt(variance);
            _invStdDev = 1.0 / _stdDev;
            _mean = mean_;
            _expNormalizer = -1.0 / (2.0 * variance);
        }

        @Override
        public double transform(double input_)
        {
            final double centered = (input_ - _mean);
            final double expArg = _expNormalizer * centered * centered;
            final double expValue = FastMath.exp(expArg);
            return expValue;
        }

        @Override
        public double derivative(int index_, double input_)
        {
            final double centered = (input_ - _mean);
            final double base = 2.0 * centered * _expNormalizer;
            final double expContribution = transform(input_);

            final double paramContribution;

            switch (index_)
            {
                case 0:
                    paramContribution = base;
                    break;
                case 1:
                    paramContribution = base * centered * _invStdDev;
                    break;
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }

            final double output = paramContribution * expContribution;
            return output;
        }

        @Override
        public double getParam(int index_)
        {
            switch (index_)
            {
                case 0:
                    return _mean;
                case 1:
                    return _stdDev; //We do this to return the original slope value. 
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }
        }
    }

    private static final class LogisticCurve extends StandardCurve<StandardCurveType>
    {
        private static final long serialVersionUID = 0x1dd40a5f2f923106L;
        private final double _center;
        private final double _slope;
        private final double _slopeParam;
        private final double _origSlope;

        public LogisticCurve(final double center_, final double slope_)
        {
            super(StandardCurveType.LOGISTIC);

            if (Double.isInfinite(center_) || Double.isNaN(center_))
            {
                throw new IllegalArgumentException("Invalid center: " + center_);
            }
            if (Double.isInfinite(slope_) || Double.isNaN(slope_))
            {
                throw new IllegalArgumentException("Invalid slope: " + slope_);
            }

            //Slope must be squared so that we can ensure that it is positive. 
            //We cannot take an abs, because that is not an analytical transformation.
            _center = center_;
            _slope = (slope_ * slope_);
            _slopeParam = Math.sqrt(_slope);
            _origSlope = slope_;
        }

        @Override
        public double transform(double input_)
        {
            final double normalized = _slope * (input_ - _center);
            final double output = MathFunctions.logisticFunction(normalized);
            return output;
        }

        @Override
        public double derivative(int index_, double input_)
        {
            final double forward = _slope * (input_ - _center);
            final double backward = -_slope * (input_ - _center);

            final double fwdLogistic = MathFunctions.logisticFunction(forward);
            final double backLogistic = MathFunctions.logisticFunction(backward);
            final double combined = fwdLogistic * backLogistic;

            final double paramContribution;

            switch (index_)
            {
                case 0:
                    paramContribution = -_slope;
                    break;
                case 1:
                    paramContribution = 2.0 * _origSlope * (input_ - _center);
                    break;
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }

            final double derivative = combined * paramContribution;
            return derivative;
        }

        @Override
        public double getParam(int index_)
        {
            switch (index_)
            {
                case 0:
                    return _center;
                case 1:
                    return _slopeParam; //We do this to return the original slope value. 
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }
        }

    }

}
