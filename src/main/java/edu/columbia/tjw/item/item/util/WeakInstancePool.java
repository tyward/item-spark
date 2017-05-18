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

import java.lang.ref.WeakReference;
import java.util.WeakHashMap;

/**
 * A pool made for deduplication of objects.
 *
 * @author tyler
 * @param <T> The type of object to be cached
 */
public final class WeakInstancePool<T>
{
    private final WeakHashMap<T, WeakReference<T>> _instanceMap;

    public WeakInstancePool()
    {
        _instanceMap = new WeakHashMap<>();
    }

    public T makeCanonical(final T input_)
    {
        if (null == input_)
        {
            return null;
        }

        final WeakReference<T> ref = _instanceMap.get(input_);

        if (null != ref)
        {
            final T val = ref.get();

            if (null != val)
            {
                return val;
            }
        }

        final WeakReference<T> newRef = new WeakReference<>(input_);
        _instanceMap.put(input_, newRef);
        return input_;
    }

    public void clear()
    {
        _instanceMap.clear();
    }
}
