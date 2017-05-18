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
package edu.columbia.tjw.fred;

import edu.columbia.tjw.item.util.HashUtil;
import java.io.IOException;
import java.io.Serializable;
import org.w3c.dom.Element;

/**
 *
 * @author tyler
 */
public final class FredCategory implements Serializable, Comparable<FredCategory>
{
    private static final long serialVersionUID = 8557088411964233149L;

    private static final int CLASS_HASH = HashUtil.startHash(FredCategory.class);

    private final String _name;
    private final int _id;
    private final int _parent;

    protected FredCategory(final Element elem_) throws IOException, FredException
    {
        final String tagName = elem_.getTagName();

        if (!tagName.equals("category"))
        {
            throw new IllegalArgumentException("Invalid element: " + tagName);
        }

        _name = elem_.getAttribute("name");

        final String idString = elem_.getAttribute("id");
        final String parentString = elem_.getAttribute("parent_id");

        _id = Integer.parseInt(idString);
        _parent = Integer.parseInt(parentString);

    }

    public boolean isRoot()
    {
        return _parent == _id;
    }

    public String getName()
    {
        return _name;
    }

    public int getId()
    {
        return _id;
    }

    public int getParentId()
    {
        return _parent;
    }

    @Override
    public int hashCode()
    {
        final int hash = HashUtil.mix(CLASS_HASH, _id);
        return hash;
    }

    @Override
    public boolean equals(final Object that_)
    {
        if (this == that_)
        {
            return true;
        }
        if (null == that_)
        {
            return false;
        }
        if (this.getClass() != that_.getClass())
        {
            return false;
        }

        final FredCategory that = (FredCategory) that_;
        final int comp = this.compareTo(that);
        final boolean equal = (0 == comp);
        return equal;
    }

    @Override
    public int compareTo(FredCategory that_)
    {
        if (null == that_)
        {
            return 1;
        }
        if (this == that_)
        {
            return 0;
        }

        return Integer.compare(this._id, that_._id);
    }

}
