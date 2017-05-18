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
package edu.columbia.tjw.item;

import java.io.Serializable;

/**
 *
 * @author tyler
 * @param <V> The type of the ITEM curve
 */
public interface ItemCurve<V extends ItemCurveType<V>>
        extends Serializable, Comparable<ItemCurve<V>>
{

    /**
     * Applies this curve to x.
     *
     * @param x_ The value to be transformed
     * @return f(x), the transformed result.
     */
    public double transform(final double x_);

    /**
     * Gets the specified parameter of this curve.
     *
     * @param index_ The index of the desired parameter
     * @return The parameter at index index_
     */
    public double getParam(final int index_);

    /**
     * Computes the derivative of this curve with respect to the specified
     * parameter, at x_
     *
     * @param index_ The index of the target parameter
     * @param x_ The point at which df/dx(x) is desired
     * @return THe derivative of f(x) at x with respect to parameter index_
     */
    public double derivative(int index_, double x_);

    /**
     * Gets the type of this curve.
     *
     * @return The type of this curve
     */
    public V getCurveType();

}
