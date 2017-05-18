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

import org.apache.commons.math3.util.FastMath;

/**
 *
 * @author tyler
 */
public final class MathFunctions
{
    public static final double EPSILON = Math.ulp(1.0);

    // SQRT of the difference between 1.0 and the smallest value larger than 1.0.
    public static final double SQRT_EPSILON = Math.sqrt(EPSILON);

    private MathFunctions()
    {
    }

    //Compute the binary logistic of the input_.
    public static double logisticFunction(final double input_)
    {
        final double exp = FastMath.exp(-input_);
        final double result = 1.0 / (1.0 + exp);
        return result;
    }

    public static double logitFunction(final double input_)
    {
        final double ratio = input_ / (1.0 - input_);
        final double output = Math.log(ratio);
        return output;
    }

    /**
     * If modelB is better, AIC diff will be less than 0.0.
     *
     * @param paramCountA_
     * @param paramCountB_
     * @param entropyA_
     * @param entropyB_
     * @param rowCount_
     * @return
     */
    public static double computeAicDifference(final int paramCountA_, final int paramCountB_, final double entropyA_, final double entropyB_, final int rowCount_)
    {
        final double llImprovement = entropyA_ - entropyB_;
        final double scaledImprovement = llImprovement * rowCount_;
        final double paramContribution = paramCountB_ - paramCountA_;
        final double aicDiff = 2.0 * (paramContribution - scaledImprovement);
        return aicDiff;
    }

    /**
     * Compares the two doubles, see if they are different beyond a level that
     * would be typical of rounding error.
     *
     * Both numbers must be well defined (not infinite, not NaN) for this to
     * work. Also, both must be positive.
     *
     * @param a_
     * @param b_
     * @return +1 if a_ &gt; b_, -1 if b_ &gt; a_, and 0 if they are too close
     * to tell.
     */
    public static int doubleCompareRounded(final double a_, final double b_)
    {
        if (!(a_ > 0.0) || Double.isInfinite(a_))
        {
            throw new IllegalArgumentException("Invalid starting value: " + a_);
        }
        if (!(b_ > 0.0) || Double.isInfinite(b_))
        {
            throw new IllegalArgumentException("Invalid ending value: " + b_);
        }

        final double diff = b_ - a_;
        final double norm = b_ + b_;
        final double relDiff = diff / norm;

        if (relDiff > SQRT_EPSILON)
        {
            return 1;
        }
        else if (relDiff < -SQRT_EPSILON)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    public static boolean isAicBetter(final double starting_, final double ending_)
    {
        //Seems fairly obvious.
        return isAicWorse(ending_, starting_);
    }

    /**
     * Is this AIC worse, beyond the minor jitter that can come from
     * non-commutative behavior of floating point math.
     *
     * @param starting_
     * @param ending_
     * @return
     */
    public static boolean isAicWorse(final double starting_, final double ending_)
    {
        return doubleCompareRounded(starting_, ending_) > 0;
    }

}
