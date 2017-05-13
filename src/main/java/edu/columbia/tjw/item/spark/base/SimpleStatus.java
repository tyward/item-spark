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

import edu.columbia.tjw.item.ItemStatus;
import static edu.columbia.tjw.item.spark.base.BinaryStatus.FAMILY;
import edu.columbia.tjw.item.util.EnumFamily;
import edu.columbia.tjw.item.util.HashUtil;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author tyler
 */
public class SimpleStatus implements ItemStatus<SimpleStatus>
{
    private static final int CLASS_HASH = HashUtil.startHash(SimpleStatus.class);
    private static final long serialVersionUID = 0x8787a642d5713061L;

    private final SimpleStringEnum _base;
    private final List<SimpleStatus> _indistinguishable;

    //These two must be filled out after construction.
    private EnumFamily<SimpleStatus> _family;
    private List<SimpleStatus> _reachable = null;

    public static EnumFamily<SimpleStatus> generateFamily(final Collection<String> regressorNames_)
    {
        final EnumFamily<SimpleStringEnum> baseFamily = SimpleStringEnum.generateFamily(regressorNames_);

        final SimpleStatus[] regs = new SimpleStatus[baseFamily.size()];
        int pointer = 0;

        for (final SimpleStringEnum next : baseFamily.getMembers())
        {
            final SimpleStatus nextReg = new SimpleStatus(next);
            regs[pointer++] = nextReg;
        }

        final EnumFamily<SimpleStatus> family = new EnumFamily<>(regs, false);

        for (final SimpleStatus reg : regs)
        {
            reg.setFamily(family);
        }

        return family;
    }

    private SimpleStatus(final SimpleStringEnum base_)
    {
        _base = base_;
        _indistinguishable = Collections.singletonList(this);
    }

    private void setFamily(final EnumFamily<SimpleStatus> family_)
    {
        _family = family_;
    }

    @Override
    public String name()
    {
        return _base.name();
    }

    @Override
    public int ordinal()
    {
        return _base.ordinal();
    }

    @Override
    public EnumFamily<SimpleStatus> getFamily()
    {
        return _family;
    }

    @Override
    public int hashCode()
    {
        return HashUtil.mix(CLASS_HASH, _base.hashCode());
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

        final SimpleStatus that = (SimpleStatus) other_;

        return this._base.equals(that._base);
    }

    @Override
    public int compareTo(final SimpleStatus that_)
    {
        if (this == that_)
        {
            return 0;
        }
        if (null == that_)
        {
            return 1;
        }

        return this._base.compareTo(that_._base);
    }

    /**
     * Assumes that each status is reachable from every other status.
     *
     * @return
     */
    @Override
    public int getReachableCount()
    {
        return _family.size();
    }

    /**
     * Assumes that there are no indistinguishable states, every status can be
     * identified in the data set.
     *
     * @return
     */
    @Override
    public List<SimpleStatus> getIndistinguishable()
    {
        return _indistinguishable;
    }

    @Override
    public List<SimpleStatus> getReachable()
    {
        if (null == _reachable)
        {
            _reachable = Collections.unmodifiableList(new ArrayList<>(_family.getMembers()));
        }

        return _reachable;
    }
}
