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

/**
 *
 * @author tyler
 */
public final class HashUtil
{
    private static final int START_CONSTANT = 2309289;
    private static final int MIX_CONSTANT = 1091349811;
    private static final int MASK = 28329;

    private HashUtil()
    {
    }

    public static int startHash(final Class<?> clazz_)
    {
        final String className = clazz_.getCanonicalName();
        final int nameHash = className.hashCode();
        final int hash = mix(START_CONSTANT, nameHash);
        return hash;
    }

    public static int mix(final int hash_, final int mixIn_)
    {
        final int hash = MASK + MIX_CONSTANT * (hash_ + mixIn_);
        return hash;
    }

    public static int mix(final int hash_, final long input_)
    {
        final int input1 = (int) (input_ & 0xFFFFFFFFL);
        final int input2 = (int) ((input_ >> 32) & 0xFFFFFFFFL);
        int hash = mix(hash_, input1);
        hash = mix(hash, input2);
        return hash;
    }

}
