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

import edu.columbia.tjw.item.ItemStatus;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author tyler
 * @param <S> The status type on which to apply this function
 */
public final class LogLikelihood<S extends ItemStatus<S>>
{
    //Nothing is less likely than 1 in a million, roughly. 
    private static final double LOG_CUTOFF = 14;
    private final double EXP_CUTOFF = Math.exp(-LOG_CUTOFF);

    //This logic assumes all indistinguishable states are adjacent.
    //private final int[][] _mapping;
    private final int[] _reachabilityMap;
    private final int[] _ordinalMap;
    private final int[][] _indistinguishableMap;

    public LogLikelihood(final S fromStatus_)
    {
        final EnumFamily<S> family = fromStatus_.getFamily();
        final int count = family.size();

        final List<S> reachable = fromStatus_.getReachable();

        _reachabilityMap = new int[count];
        _ordinalMap = new int[reachable.size()];
        _indistinguishableMap = new int[reachable.size()][];

        Arrays.fill(_reachabilityMap, -1);

        for (int i = 0; i < reachable.size(); i++)
        {
            final S next = reachable.get(i);
            final int nextOrdinal = next.ordinal();
            _ordinalMap[i] = nextOrdinal;
            _reachabilityMap[nextOrdinal] = i;

        }

        for (int i = 0; i < reachable.size(); i++)
        {
            final S next = reachable.get(i);
            final List<S> indistinguishable = next.getIndistinguishable();
            final int indCount = indistinguishable.size();

            _indistinguishableMap[i] = new int[indCount];
            int indPointer = 0;

            //N.B: Watch carefully. If two (or more) states are indistinguishable, and we can get to both of them...
            //Then for every one we can reach, add it to the indistinguishable set. This set will always contain at least one
            //state (namely, the state itself). 
            for (int w = 0; w < indCount; w++)
            {
                final int indOrdinal = indistinguishable.get(w).ordinal();
                final int mapped = _reachabilityMap[indOrdinal];

                if (mapped >= 0)
                {
                    _indistinguishableMap[i][indPointer++] = mapped;
                }
            }

            _indistinguishableMap[i] = Arrays.copyOf(_indistinguishableMap[i], indPointer);
        }
    }

    /**
     * Computes the ordinal corresponding to the given offset.
     *
     * Not every status can transition to every other status. The available
     * transitions are packed tightly, omitting any impossible transitions.
     * Therefore, an offset into the output array won't necessarily be the
     * ordinal of the status represented. This function performs that mapping.
     *
     * @param offset_ The offset
     * @return The ordinal corresponding to the offset
     */
    public final int offsetToOrdinal(final int offset_)
    {
        return _ordinalMap[offset_];
    }

    /**
     * Returns the expected offset into the computed array of the given ordinal.
     *
     * Returns -1 if this ordinal is not reachable from this state.
     *
     * @param ordinal_ the ordinal
     * @return The offset
     */
    public final int ordinalToOffset(final int ordinal_)
    {
        return _reachabilityMap[ordinal_];
    }

    public final double logLikelihood(final double[] computed_, final int actualTransitionOffset_)
    {
        if (actualTransitionOffset_ < 0)
        {
            //N.B: The data included a forbidden transition, we need to ignore that data point.
            return 0.0;
        }

        double computed = 0.0;
        final int[] indistinguishable = _indistinguishableMap[actualTransitionOffset_];

        for (int i = 0; i < indistinguishable.length; i++)
        {
            final int next = indistinguishable[i];
            computed += computed_[next];
        }

        final double output = logLikelihoodTerm(1.0, computed);
        return output;
    }

    private double logLikelihoodTerm(final double actual_, final double computed_)
    {
        if (actual_ <= 0.0)
        {
            return 0.0;
        }

        final double negLL;

        if (computed_ > EXP_CUTOFF)
        {
            negLL = -Math.log(computed_);
        }
        else
        {
            negLL = LOG_CUTOFF;
        }

        final double product = actual_ * negLL;
        return product;
    }

}
