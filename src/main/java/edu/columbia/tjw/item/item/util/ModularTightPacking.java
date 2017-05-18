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

import java.util.Arrays;

/**
 * Tightly pack the given input array using modular arithmetic.
 *
 * Given an array R containing N integers, can we compute A and B such that the
 * operation Q = ((x-A) mod B) will produce the integers [0, N) when given the
 * original contents of R. I believe this is always possible, but the code
 * checks this assumption.
 *
 * When this is possible, then the values of R can be packed into a new array Y
 * indexed by Q, and in this way the operation can also be checked. If given an
 * integer x not in R, the index Q will be computed, but Y[Q] != x, and an
 * exception is thrown.
 *
 * Combined with an array of type T, this can function as a Map&lt;int, T&gt;,
 * but at only a tiny fraction of the cost.
 *
 * @author tyler
 */
public final class ModularTightPacking
{
    private final int _modulus;
    private final int _min;
    private final int[] _packedArray;

    public ModularTightPacking(final int[] input_)
    {
        final int[] workspace = input_.clone();

        Arrays.sort(workspace);

        final int arraySize = workspace.length;
        int min = workspace[0];
        int max = workspace[0];

        for (int i = 1; i < arraySize; i++)
        {
            if (workspace[i - 1] == workspace[i])
            {
                throw new IllegalArgumentException("Inputs must be unique.");
            }

            min = Math.min(workspace[i], min);
            max = Math.max(workspace[i], max);
        }

        _min = min;
        final int maxSize = max - min;
        int modulus = 0;

        for (int i = 0; i < arraySize; i++)
        {
            workspace[i] = workspace[i] - min;
        }

        for (int i = input_.length; i < maxSize; i++)
        {
            if (testPacking(i, workspace))
            {
                modulus = i;
                break;
            }
        }

        if (0 == modulus)
        {
            throw new IllegalArgumentException("Impossible.");
        }

        _modulus = modulus;

        _packedArray = new int[modulus];

        for (int i = 0; i < arraySize; i++)
        {
            final int val = input_[i];
            final int index = computeIndex(val, false);
            _packedArray[index] = val;
        }

    }

    private static boolean testPacking(final int modulus_, final int[] workspace_)
    {
        final boolean[] testArray = new boolean[modulus_];

        for (int i = 0; i < workspace_.length; i++)
        {
            final int testIndex = workspace_[i] % modulus_;

            if (testArray[testIndex])
            {
                return false;
            }

            testArray[testIndex] = true;
        }

        return true;
    }

    public int getArraySize()
    {
        return _modulus;
    }

    /**
     * Computes the index Q corresponding to an element of R. If the given
     * input_ is not an element of R, returns -1.
     *
     * @param input_ The element of R to convert to an index Q
     * @return Q if input_ is an element of R, otherwise -1
     */
    public int computeIndex(final int input_)
    {
        return computeIndex(input_, true);
    }

    /**
     * Returns the index Q extracted from input_.
     *
     * @param input_ The input to convert.
     * @param withCheck_ True if this operation should check for invalid
     * lookups.
     * @return Q if input_ is an element of R, otherwise undefined.
     */
    private int computeIndex(final int input_, final boolean withCheck_)
    {
        final int index = (input_ - _min) % _modulus;

        if (withCheck_)
        {
            final int checkVal = _packedArray[index];

            if (checkVal != input_)
            {
                return -1;
            }
        }

        return index;
    }

}
