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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 *
 * This is needed because many of the relevant methods in enum (e.g. values())
 * are static, so they cannot be used against objects of unknown types.
 *
 * @author tyler
 * @param <V> The type of enum that composes this family.
 */
public final class EnumFamily<V extends EnumMember<V>> implements Serializable
{
    private static final long serialVersionUID = 2720474101494526203L;

    private static final Map<Class, EnumFamily> FAMILY_MAP = new HashMap<>();

    private final V[] _members;
    private final SortedSet<V> _memberSet;
    private final Map<String, V> _nameMap;
    private final Class<? extends V> _componentClass;

    @SuppressWarnings("unchecked")
    public static <V extends EnumMember<V>> EnumFamily<V> getFamilyFromClass(final Class<V> familyClass_, final boolean throwOnMissing_)
    {
        synchronized (FAMILY_MAP)
        {
            final EnumFamily<V> result = (EnumFamily<V>) FAMILY_MAP.get(familyClass_);

            if (null == result && throwOnMissing_)
            {
                throw new IllegalArgumentException("No family for class: " + familyClass_);
            }

            return result;
        }
    }

    public EnumFamily(final V[] values_)
    {
        this(values_, true);
    }

    /**
     * Initialize a new enum family, should pass it enum.values().
     *
     * @param values_ The output of enum.values() should be given here.
     * @param distinctFamily_ True if this class should only have one associated
     * EnumFamily.
     */
    @SuppressWarnings("unchecked")
    public EnumFamily(final V[] values_, final boolean distinctFamily_)
    {
        if (values_.length < 1)
        {
            throw new IllegalArgumentException("Values must have positive length.");
        }
        for (final V value : values_)
        {
            if (null == value)
            {
                throw new NullPointerException("Enum members cannot be null.");
            }
        }

        _members = values_.clone();
        _memberSet = Collections.unmodifiableSortedSet(new TreeSet<>(Arrays.asList(_members)));

        if (_members.length != _memberSet.size())
        {
            throw new IllegalArgumentException("Members are not distinct!");
        }

        _nameMap = new HashMap<>();
        int pointer = 0;

        for (final V next : _memberSet)
        {
            if ((_members[pointer].ordinal() != pointer) || (next != _members[pointer++]))
            {
                throw new IllegalArgumentException("Members out of order.");
            }

            _nameMap.put(next.name(), next);
        }

        //we actually know that this cast is valid, provided values is actually of type V. 
        _componentClass = (Class<? extends V>) _members[0].getClass();

        if (distinctFamily_)
        {
            synchronized (FAMILY_MAP)
            {
                if (FAMILY_MAP.containsKey(_componentClass))
                {
                    throw new IllegalArgumentException("Attempt to redefine an enum family.");
                }

                FAMILY_MAP.put(_componentClass, this);
            }
        }
    }

    /**
     * Returns the class of the component members of this family.
     *
     * @return The class of the members of this family.
     */
    public Class<? extends V> getComponentType()
    {
        return _componentClass;
    }

    /**
     * Returns all members of this enum family.
     *
     * @return Same as enum.values()
     */
    public SortedSet<V> getMembers()
    {
        return _memberSet;
    }

    /**
     * How many members does this enum have.
     *
     * @return enum.values().length
     */
    public int size()
    {
        return _members.length;
    }

    /**
     * Look up an enum by its ordinal
     *
     * @param ordinal_ The ordinal of the enum to retrieve.
     * @return enum.values()[ordinal_]
     */
    public V getFromOrdinal(final int ordinal_)
    {
        return _members[ordinal_];
    }

    /**
     * Look up an enum by name.
     *
     * @param name_ The name of the enum to look up.
     * @return enum.fromName(name_)
     */
    public V getFromName(final String name_)
    {
        return _nameMap.get(name_);
    }

    private Object readResolve()
    {
        //Lots of odd ordering issues can happen due to serialization, be 
        //tolerant of things that are half built or otherwise problematic.
        for (final V next : this._members)
        {
            if (null == next)
            {
                continue;
            }

            final EnumFamily<V> fam = next.getFamily();

            if (null != fam)
            {
                return fam;
            }
        }

        //Hmm, looks like the members are still being initialized, just return 
        //this object as-is, it will likely be used to fill out the member 
        //family objects.
        return this;
    }

    /**
     * Generates a new array of the type of the enum members with the given
     * size.
     *
     * All elements are initially set to null.
     *
     * @param size_ The size of the returned array.
     * @return A new empty array of type V with size size_
     */
    public V[] generateTypedArray(final int size_)
    {
        final V[] output = Arrays.copyOf(_members, size_);
        Arrays.fill(output, null);
        return output;
    }

}
