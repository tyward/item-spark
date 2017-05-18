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
package edu.columbia.tjw.item.data;

import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.util.EnumFamily;

/**
 *
 * @author tyler
 * @param <R> The type of the regressor represented by this grid
 */
public interface ItemGrid<R extends ItemRegressor<R>>
{

    public boolean hasRegressorReader(final R field_);

    /**
     * Gets the reader for the given field
     *
     * @param field_ The regressor to fetch
     * @return A reader used to get data from this column
     */
    public ItemRegressorReader getRegressorReader(final R field_);

    public int size();

    public EnumFamily<R> getRegressorFamily();

}
