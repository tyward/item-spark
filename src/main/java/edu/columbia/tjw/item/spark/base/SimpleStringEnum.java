/*
 * Copyright 2017 tyler.
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
 */
package edu.columbia.tjw.item.spark.base;

import edu.columbia.tjw.item.util.EnumFamily;
import edu.columbia.tjw.item.util.EnumMember;
import edu.columbia.tjw.item.util.HashUtil;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * Constructs a proper EnumFamily from a collection of strings.
 *
 * It is slightly better to define the EnumMembers as an actual java enum (see
 * BinaryStatus), because that ensures that each family is a singleton. As is,
 * this code makes reasonable attempts to make the family a fairly coherent
 * singleton but it can't be guaranteed.
 *
 *
 * @author tyler
 */
public final class SimpleStringEnum implements EnumMember<SimpleStringEnum>
{
    private static final int CLASS_HASH = HashUtil.startHash(SimpleStringEnum.class);
    private static final long serialVersionUID = 0x2e3ad6fa3cd2b465L;

    private final int _ordinal;
    private final String _name;
    private final int _hashCode;
    private EnumFamily<SimpleStringEnum> _family;

    public static EnumFamily<SimpleStringEnum> generateFamily(final Collection<String> regressorNames_)
    {
        if (regressorNames_.isEmpty())
        {
            throw new IllegalArgumentException("Cannot create an empty regressor set.");
        }

        final int size = regressorNames_.size();
        final Set<String> checkSet = new HashSet<>(regressorNames_);

        if (checkSet.size() != size)
        {
            throw new IllegalArgumentException("Regressor names are not distinct.");
        }

        final SimpleStringEnum[] regs = new SimpleStringEnum[size];
        int pointer = 0;
        final int hashBase = checkSet.hashCode();

        for (final String next : regressorNames_)
        {
            final SimpleStringEnum nextReg = new SimpleStringEnum(pointer, next, hashBase);
            regs[pointer++] = nextReg;
        }

        final EnumFamily<SimpleStringEnum> family = new EnumFamily<>(regs, false);

        for (final SimpleStringEnum reg : regs)
        {
            reg.setFamily(family);
        }

        return family;
    }

    private SimpleStringEnum(final int ordinal_, final String name_, final int hashBase_)
    {
        _ordinal = ordinal_;
        _name = name_;

        int hash = HashUtil.mix(CLASS_HASH, hashBase_);
        hash = HashUtil.mix(hash, name_.hashCode());
        _hashCode = HashUtil.mix(hash, ordinal_);
    }

    private void setFamily(final EnumFamily<SimpleStringEnum> family_)
    {
        _family = family_;
    }

    @Override
    public String name()
    {
        return _name;
    }

    @Override
    public int ordinal()
    {
        return _ordinal;
    }

    @Override
    public EnumFamily<SimpleStringEnum> getFamily()
    {
        return _family;
    }

    @Override
    public int hashCode()
    {
        return _hashCode;
    }

    @Override
    public boolean equals(final Object other_)
    {
        if (this == other_)
        {
            return true;
        }
        if (null == other_)
        {
            return false;
        }
        if (this.getClass() != other_.getClass())
        {
            return false;
        }

        final SimpleStringEnum that = (SimpleStringEnum) other_;

        if (this.getFamily() != that.getFamily())
        {
            return false;
        }

        return (0 == this.compareTo(that));
    }

    @Override
    public int compareTo(final SimpleStringEnum that_)
    {
        if (this == that_)
        {
            return 0;
        }
        if (null == that_)
        {
            return 1;
        }

        if (this.getFamily() != that_.getFamily())
        {
            throw new IllegalArgumentException("Incomparable families: " + this.getFamily() + " != " + that_.getFamily());
        }

        //Within a family, compare based on ordinal.
        return Integer.compare(this.ordinal(), that_.ordinal());
    }
}
