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


import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.util.EnumFamily;

/**
 *
 * @author tyler
 */
public enum SplineCurveType implements ItemCurveType<SplineCurveType>
{
    STEP(2),
    BASIS(2);

    public static final EnumFamily<SplineCurveType> FAMILY = new EnumFamily<>(values());

    private final int _paramCount;

    private SplineCurveType(final int paramCount_)
    {
        _paramCount = paramCount_;
    }

    @Override
    public int getParamCount()
    {
        return _paramCount;
    }

    @Override
    public EnumFamily<SplineCurveType> getFamily()
    {
        return FAMILY;
    }
}
