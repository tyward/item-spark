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

import edu.columbia.tjw.item.ItemModel;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.util.random.RandomTool;
import java.util.List;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;

/**
 *
 * @author tyler
 * @param <S> Status being modeled
 * @param <R> Type of regressors used
 */
public class ItemClassificationModel<S extends ItemStatus<S>, R extends ItemRegressor<R>> extends ProbabilisticClassificationModel<Vector, ItemClassificationModel<S, R>>
{
    private static final long serialVersionUID = 0x8c7eb061e0d2980aL;

    private final ItemParameters<S, R, StandardCurveType> _params;
    private final int[] _offsetMap;
    private String _uid;

    private transient ItemModel<S, R, StandardCurveType> _model;
    private transient double[] _rawRegressors;

    public ItemClassificationModel(final ItemParameters<S, R, StandardCurveType> params_, final List<R> fieldOrdering_)
    {
        _params = params_;

        final List<R> paramFields = params_.getUniqueRegressors();

        _offsetMap = new int[paramFields.size()];

        //This is the intercept, always. 
        _offsetMap[0] = -1;

        for (int i = 1; i < paramFields.size(); i++)
        {
            final R next = paramFields.get(i);

            if (next == _params.getEntryRegressor(_params.getInterceptIndex(), 0))
            {
                _offsetMap[i] = -1;
                continue;
            }

            final int index = fieldOrdering_.indexOf(next);

            if (-1 == index)
            {
                throw new IllegalArgumentException("Missing regressors from fields.");
            }

            _offsetMap[i] = index;
        }
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
        final ItemModel<S, R, StandardCurveType> model = getModel();

        for (int i = 0; i < _params.getUniqueRegressors().size(); i++)
        {
            final int fieldIndex = _offsetMap[i];

            if (-1 == fieldIndex)
            {
                _rawRegressors[i] = 1.0;
            }
            else
            {
                _rawRegressors[i] = allRegressors_.apply(fieldIndex);
            }
        }

        final double[] probabilities = new double[_params.getStatus().getReachableCount()];
        model.transitionProbability(_rawRegressors, probabilities);
        return new DenseVector(probabilities);
    }

    @Override
    public ItemClassificationModel<S, R> copy(ParamMap arg0)
    {
        return defaultCopy(arg0);
    }

    @Override
    public synchronized String uid()
    {
        //Hideous hack because this is being called in the superclass constructor.
        if (null == _uid)
        {
            _uid = RandomTool.randomString(64);
        }

        return _uid;
    }

    public final ItemParameters<S, R, StandardCurveType> getParams()
    {
        return _params;
    }

    private ItemModel<S, R, StandardCurveType> getModel()
    {
        if (null == _model)
        {
            _model = new ItemModel<>(_params);
            _rawRegressors = new double[_params.getUniqueRegressors().size()];
        }

        return _model;
    }

}
