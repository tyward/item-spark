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
import java.util.Random;

/**
 *
 * @author tyler
 * @param <R>
 */
public class SplineFactory<R extends ItemRegressor<R>> implements ItemCurveFactory<R, SplineCurveType>
{
    /**
     * The singleton for this class. It has no free parameters, so no need for
     * more than one.
     */
    public static final SplineFactory SINGLETON = new SplineFactory();

    private SplineFactory()
    {
    }

    @Override
    public ItemCurve<SplineCurveType> generateCurve(SplineCurveType type_, int offset_, double[] params_)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public ItemCurveParams<R, SplineCurveType> generateStartingParameters(final SplineCurveType type_, final R field_, final QuantileDistribution dist_, final Random rand_)
    {
        throw new UnsupportedOperationException("Not supported.");
//
//        final double[] curveParams = new double[2]; //== type_.getParamCount();
//        curveParams[0] = mean_;
//
//        switch (type_)
//        {
//            case STEP:
//                curveParams[1] = Math.sqrt(1.0 / (stdDev_ + 1.0e-10));
//                break;
//            case BASIS:
//                curveParams[1] = stdDev_;
//                break;
//            default:
//                throw new RuntimeException("Impossible.");
//        }
//
//        final double beta = 0.0;
//        final double intercept = 0.0;
//
//        final ItemCurveParams<SplineCurveType> output = new ItemCurveParams<>(type_, intercept, beta, curveParams);
//
//        throw new UnsupportedOperationException("Not supported.");
//        //return output;
    }

    @Override
    public EnumFamily<SplineCurveType> getFamily()
    {
        return SplineCurveType.FAMILY;
    }

    @Override
    public ItemCurve<SplineCurveType> boundCentrality(ItemCurve<SplineCurveType> inputCurve_, double lowerBound_, double upperBound_)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private static final class BasisSpline extends StandardCurve<SplineCurveType>
    {
        private final double _center;
        private final double _radius;
        private final double _stdDev;
        private final double _radParam;

        public BasisSpline(final double mean_, final double stdDev_)
        {
            super(SplineCurveType.BASIS);

            if (Double.isInfinite(mean_) || Double.isNaN(mean_))
            {
                throw new IllegalArgumentException("Invalid mean: " + mean_);
            }
            if (Double.isInfinite(stdDev_) || Double.isNaN(stdDev_))
            {
                throw new IllegalArgumentException("Invalid stdDev: " + stdDev_);
            }

            final double variance = (stdDev_ * stdDev_) + 1.0e-10;
            _center = mean_;
            _radius = variance;
            _stdDev = Math.sqrt(variance);
            _radParam = stdDev_;
        }

        @Override
        public double transform(double input_)
        {
            final double distance = Math.abs(input_ - _center);
            final double capped = Math.min(distance, _radius);
            final double scaled = capped / _radius;
            final double output = 1.0 - scaled;
            return output;
        }

        @Override
        public double derivative(int index_, double input_)
        {
            final double distance = Math.abs(input_ - _center);

            if (distance > _radius)
            {
                return 0.0;
            }

            if (index_ == 0)
            {
                if (input_ >= _center)
                {
                    return 1.0 / _radius;
                }
                else
                {
                    return -1.0 / _radius;
                }
            }

            return distance / (_radius * _radius);
        }

        @Override
        public double getParam(int index_)
        {
            switch (index_)
            {
                case 0:
                    return _center;
                case 1:
                    return _radParam; //We do this to return the original slope value. 
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }
        }
    }

    private static final class StepSpline extends StandardCurve<SplineCurveType>
    {
        private final double _start;
        private final double _end;
        private final double _width;
        private final double _widthParam;

        public StepSpline(final double center_, final double width_)
        {
            super(SplineCurveType.STEP);

            if (Double.isInfinite(center_) || Double.isNaN(center_))
            {
                throw new IllegalArgumentException("Invalid center: " + center_);
            }
            if (Double.isInfinite(width_) || Double.isNaN(width_))
            {
                throw new IllegalArgumentException("Invalid slope: " + width_);
            }

            _widthParam = Math.abs(width_);
            _width = (width_ * width_) + 1.0e-10;
            _start = center_;
            _end = _start + _width;

        }

        @Override
        public double transform(double input_)
        {
            if (input_ <= _start)
            {
                return 0.0;
            }
            if (input_ >= _end)
            {
                return 1.0;
            }

            final double output = (input_ - _start) / _width;
            return output;
        }

        @Override
        public double derivative(int index_, double input_)
        {
            if (input_ <= _start || input_ >= _end)
            {
                return 0.0;
            }

            switch (index_)
            {
                case 0:
                    return -1.0 / _width;
                case 1:
                    final double derivative = -(index_ - _start) / (_width * _width);
                    return derivative;
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }

        }

        @Override
        public double getParam(int index_)
        {
            switch (index_)
            {
                case 0:
                    return _start;
                case 1:
                    return _widthParam; //We do this to return the original slope value. 
                default:
                    throw new IllegalArgumentException("Bad index: " + index_);
            }
        }

    }

}
