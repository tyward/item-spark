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
package edu.columbia.tjw.item.optimize;

/**
 *
 * @author tyler
 * @param <V> The type of this EvaluationPoint
 */
public interface EvaluationPoint<V extends EvaluationPoint<V>>
        extends Cloneable
{

    /**
     * Returns the scalar that when used to scalar multiply input_, produces the projection of this onto input.
     * 
     * @param input_ The point upon which this point should be projected
     * @return The scalar projection of this onto input_
     */
    public double project(final V input_);
    
    public double getMagnitude();

    public double distance(final V point_);

    public void scale(final double input_);

    public void copy(final V point_);
    
    public void add(final V point_);
    
    public void normalize();

    public V clone();

}
