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
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public class ItemClassificationModel<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> extends ProbabilisticClassificationModel<Vector, ItemClassificationModel<S, R, T>>
{
    private static final long serialVersionUID = 0x8c7eb061e0d2980aL;

    private final ItemParameters<S, R, T> _params;
    private final String _uid;

    private transient ItemModel<S, R, T> _model;
    private transient double[] _rawRegressors;

    public ItemClassificationModel(final ItemParameters<S, R, T> params_)
    {
        _params = params_;
        _uid = RandomTool.randomString(64);
    }

    @Override
    public Vector raw2probabilityInPlace(final Vector rawProbabilities_)
    {
        //Do nothing, these are already probabilities.
        return rawProbabilities_;
    }

    @Override
    public int numClasses()
    {
        return _params.getStatus().getReachableCount();
    }

    @Override
    public Vector predictRaw(final Vector allRegressors_)
    {
        final ItemModel<S, R, T> model = getModel();

        int pointer = 0;

        for (final R next : _params.getUniqueRegressors())
        {
            _rawRegressors[pointer++] = allRegressors_.apply(next.ordinal());
        }

        final double[] probabilities = new double[_params.getStatus().getReachableCount()];

        model.transitionProbability(_rawRegressors, probabilities);

        return new DenseVector(probabilities);
    }

    @Override
    public ItemClassificationModel<S, R, T> copy(ParamMap arg0)
    {
        //I'm really not sure what this is supposed to accomplish.
        return this;
    }

    @Override
    public String uid()
    {
        return _uid;
    }

    public final ItemParameters<S, R, T> getParams()
    {
        return _params;
    }

    private ItemModel<S, R, T> getModel()
    {
        if (null == _model)
        {
            _model = new ItemModel<>(_params);
            _rawRegressors = new double[_params.getUniqueRegressors().size()];
        }

        return _model;
    }

}
