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
import java.util.Collections;
import java.util.List;

/**
 *
 * @author tyler
 */
public final class ListTool
{
    private ListTool()
    {
    }

    @SafeVarargs
    public static final <T> List<T> listify(final Class<T> type_, final T... data_)
    {
        if (data_.getClass().getComponentType() != type_)
        {
            throw new IllegalArgumentException("Type mismatch.");
        }

        final List<T> raw = Arrays.asList(data_);
        final List<T> output = Collections.unmodifiableList(raw);
        return output;
    }

}
