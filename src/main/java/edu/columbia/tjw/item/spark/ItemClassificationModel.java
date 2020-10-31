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
import edu.columbia.tjw.item.fit.FitResult;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;

import java.io.*;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author tyler
 */
public class ItemClassificationModel extends ProbabilisticClassificationModel<Vector, ItemClassificationModel>
{
    private static final long serialVersionUID = 0x2bb4508735311c26L;

    private final int[] _offsetMap;
    private final ItemClassifierSettings _settings;
    private final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> _fitResult;
    private String _uid;

    private transient ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> _model;
    private transient double[] _rawRegressors;

    public ItemClassificationModel(final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult_,
                                   final ItemClassifierSettings settings_)
    {
        _fitResult = fitResult_;
        _settings = settings_;

        final List<SimpleRegressor> paramFields = getParams().getUniqueRegressors();

        _offsetMap = new int[paramFields.size()];

        for (int i = 1; i < paramFields.size(); i++)
        {
            final SimpleRegressor next = paramFields.get(i);
            final int index = _settings.getRegressors().indexOf(next);

            if (-1 == index)
            {
                throw new IllegalArgumentException("Missing regressors from fields.");
            }

            _offsetMap[i] = index;
        }
    }

    public ItemClassifierSettings getSettings()
    {
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
        return getParams().getStatus().getReachableCount();
    }

    @Override
    public Vector predictRaw(final Vector allRegressors_)
    {
        final ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> model = getModel();

        for (int i = 0; i < getParams().getUniqueRegressors().size(); i++)
        {
            final int fieldIndex = _offsetMap[i];
            _rawRegressors[i] = allRegressors_.apply(fieldIndex);
        }

        final double[] probabilities = new double[getParams().getStatus().getReachableCount()];
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


    public FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> getFitResult()
    {
        return _fitResult;
    }

    public final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> getParams()
    {
        return getFitResult().getParams();
    }

    private ItemModel<SimpleStatus, SimpleRegressor, StandardCurveType> getModel()
    {
        if (null == _model)
        {
            _model = new ItemModel<>(getParams());
            _rawRegressors = new double[getParams().getUniqueRegressors().size()];
        }

        return _model;
    }

    public void save(final String fileName_) throws IOException
    {
        try (final FileOutputStream fout = new FileOutputStream(fileName_);
             final GZIPOutputStream zipOut = new GZIPOutputStream(fout);
             final ObjectOutputStream oOut = new ObjectOutputStream(zipOut);)
        {
            oOut.writeObject(this);
        }
    }


    public static ItemClassificationModel load(final String filename_) throws IOException
    {
        try (final FileInputStream fIn = new FileInputStream(filename_);
             final GZIPInputStream zipIn = new GZIPInputStream(fIn);
             final ObjectInputStream oIn = new ObjectInputStream(zipIn))
        {

            return (ItemClassificationModel) oIn.readObject();
        }
        catch (ClassNotFoundException e)
        {
            throw new IOException("Unable to load unknown class.", e);
        }


    }
}
