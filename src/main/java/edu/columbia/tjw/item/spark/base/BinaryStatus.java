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
import edu.columbia.tjw.item.util.EnumFamily;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This is a simple binary status. Generally, if trying to predict a binary
 * classification problem, set STATUS_A as the base ("fromStatus"), and then the
 * model will calculate the probabilities of STATUS_A and STATUS_B. By making
 * STATUS_A the base, positive beta values will indicate higher probability of
 * STATUS_B. If one status is much more likely than the other, making it the
 * base can make the resulting parameters slightly easier to interpret. This is
 * not a requirement, the code will work just fine either way.
 *
 * @author tyler
 */
public enum BinaryStatus implements ItemStatus<BinaryStatus>
{
    STATUS_A,
    STATUS_B;

    private final List<BinaryStatus> _indistinguishable;
    private List<BinaryStatus> _reachable = null;

    private BinaryStatus()
    {
        _indistinguishable = Collections.singletonList(this);
    }

    public static final EnumFamily<BinaryStatus> FAMILY = new EnumFamily<>(values());

    @Override
    public EnumFamily<BinaryStatus> getFamily()
    {
        return FAMILY;
    }

    @Override
    public int getReachableCount()
    {
        return FAMILY.size();
    }

    @Override
    public List<BinaryStatus> getIndistinguishable()
    {
        return _indistinguishable;
    }

    @Override
    public List<BinaryStatus> getReachable()
    {
        if (null == _reachable)
        {
            _reachable = Collections.unmodifiableList(new ArrayList<>(FAMILY.getMembers()));
        }

        return _reachable;
    }

}
