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

import java.io.Serializable;

/**
 * A base class for enumerations.
 *
 * This allows for the efficient enumeration of the ordinal members, for
 * instance.
 *
 * This is needed because many of the relevant methods in enum (e.g. values())
 * are static, so they cannot be used against objects of unknown types.
 *
 *
 * @author tyler
 * @param <V> The type of this enum member.
 */
public interface EnumMember<V extends EnumMember<V>> extends Comparable<V>, Serializable
{
    /**
     * Same as enum.name()
     *
     * @return enum.name()
     */
    public String name();

    /**
     * Same as enum.ordinal()
     *
     * @return enum.ordinal()
     */
    public int ordinal();

    /**
     * Return the family describing all members of this enum class.
     *
     * @return The family describing all members of this enum.
     */
    public EnumFamily<V> getFamily();

}
