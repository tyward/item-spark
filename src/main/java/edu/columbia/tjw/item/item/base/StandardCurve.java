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
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.util.HashUtil;

/**
 *
 * @author tyler
 * @param <V> The type of this family of curves
 */
public abstract class StandardCurve<V extends ItemCurveType<V>> implements ItemCurve<V>
{
    private static final long serialVersionUID = 1056495046007095245L;
    private final V _type;

    public StandardCurve(final V type_)
    {
        _type = type_;
    }

    @Override
    public final V getCurveType()
    {
        return _type;
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();

        builder.append(this.getClass());
        builder.append("[");
        builder.append(this.getCurveType());
        builder.append("]: ");

        final int paramCount = this.getCurveType().getParamCount();

        for (int i = 0; i < paramCount; i++)
        {
            if (i != 0)
            {
                builder.append(", ");
            }

            builder.append(this.getParam(i));
        }

        return builder.toString();
    }

    @Override
    public final int compareTo(final ItemCurve<V> otherCurve_)
    {
        if (null == otherCurve_)
        {
            return 1;
        }
        if (this == otherCurve_)
        {
            return 0;
        }

        if (!(otherCurve_ instanceof StandardCurve))
        {
            //This is OK, since compareTo is final.
            throw new IllegalArgumentException("Types not comparable.");
        }

        final StandardCurve<V> that = (StandardCurve<V>) otherCurve_;

        final int typeCompare = this.getCurveType().compareTo(that.getCurveType());

        if (typeCompare != 0)
        {
            return typeCompare;
        }

        final int size = this.getCurveType().getParamCount();

        for (int i = 0; i < size; i++)
        {
            final Double d1 = this.getParam(i);
            final Double d2 = that.getParam(i);

            final int comp = Double.compare(d1, d2);

            if (comp != 0)
            {
                return comp;
            }
        }

        return 0;
    }

    @Override
    public final boolean equals(final Object other_)
    {
        if (null == other_)
        {
            return false;
        }
        if (this == other_)
        {
            return true;
        }
        if (this.getClass() != other_.getClass())
        {
            return false;
        }
        final StandardCurve<?> that = (StandardCurve<?>) other_;
        if (this.getCurveType() != that.getCurveType())
        {
            return false;
        }
        final int size = this.getCurveType().getParamCount();
        for (int i = 0; i < size; i++)
        {
            final Double d1 = this.getParam(i);
            final Double d2 = that.getParam(i);
            //Compare as longs, since that's how we hashed them. THis is not exactly the same as comparing doubles directly.
            final long l1 = Double.doubleToLongBits(d1);
            final long l2 = Double.doubleToLongBits(d2);
            if (l1 != l2)
            {
                return false;
            }
        }
        return true;
    }

    @Override
    public final int hashCode()
    {
        int hash = HashUtil.startHash(this.getClass());
        hash = HashUtil.mix(hash, this.getCurveType().hashCode());
        final int paramCount = this.getCurveType().getParamCount();
        hash = HashUtil.mix(hash, paramCount);
        for (int i = 0; i < paramCount; i++)
        {
            final long longBits = Double.doubleToLongBits(this.getParam(i));
            hash = HashUtil.mix(hash, longBits);
        }
        return hash;
    }

}
