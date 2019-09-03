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
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;

import java.io.*;
import java.util.List;

/**
 * @author tyler
 */
public class ItemClassificationModel extends ProbabilisticClassificationModel<Vector, ItemClassificationModel>
{
    private static final long serialVersionUID = 0x8c7eb061e0d2980aL;

    private final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> _params;
    private final int[] _offsetMap;
    private final ItemClassifierSettings _settings;
    private String _uid;

    private transient ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> _model;
    private transient double[] _rawRegressors;

    public ItemClassificationModel(final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> params_, final ItemClassifierSettings settings_)
    {
        _params = params_;
        _settings = settings_;

        final List<SimpleRegressor> paramFields = params_.getUniqueRegressors();

        _offsetMap = new int[paramFields.size()];

        //This is the intercept, always. 
        _offsetMap[0] = -1;

        for (int i = 1; i < paramFields.size(); i++)
        {
            final SimpleRegressor next = paramFields.get(i);

            if (next == _params.getEntryRegressor(_params.getInterceptIndex(), 0))
            {
                _offsetMap[i] = -1;
                continue;
            }

            final int index = _settings.getRegressors().indexOf(next);

            if (-1 == index)
            {
                throw new IllegalArgumentException("Missing regressors from fields.");
            }

            _offsetMap[i] = index;
        }
    }

    public ItemClassifierSettings getSettings() {
        return _settings;
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
        final ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> model = getModel();

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
    public ItemClassificationModel copy(ParamMap arg0)
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

    public final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> getParams()
    {
        return _params;
    }

    private ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> getModel()
    {
        if (null == _model)
        {
            _model = new ItemModel<>(_params);
            _rawRegressors = new double[_params.getUniqueRegressors().size()];
        }

        return _model;
    }

    public void save(final String fileName_) throws IOException
    {
        try (final FileOutputStream fout = new FileOutputStream(fileName_);
             final ObjectOutputStream oOut = new ObjectOutputStream(fout);)
        {
            oOut.writeObject(this);


        }
    }


    public static ItemClassificationModel load(final String filename_) throws IOException
    {
        try (final FileInputStream fIn = new FileInputStream(filename_);
             final ObjectInputStream oIn = new ObjectInputStream(fIn))
        {

            return (ItemClassificationModel) oIn.readObject();
        }
        catch (ClassNotFoundException e)
        {
            throw new IOException("Unable to load unknown class.", e);
        }


    }
}
