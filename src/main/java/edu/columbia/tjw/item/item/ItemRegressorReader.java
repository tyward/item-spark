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

/**
 * A thin wrapper used to read specific regressors from the model data.
 *
 * This is used by the curve drawing logic.
 *
 * @author tyler
 */
public interface ItemRegressorReader
{
    /**
     * Get the regressor for observation index_
     *
     * Must support index values from 0 to modelGrid.size(), exclusive.
     *
     * @param index_ The index of the observation
     * @return The value of the regressor for that observation.
     */
    public double asDouble(final int index_);
    
    /**
     * Returns the number of elements in this array.
     * @return The number of elements in this reader
     */
    public int size();
    

}
