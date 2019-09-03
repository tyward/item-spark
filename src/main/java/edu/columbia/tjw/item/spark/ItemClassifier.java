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

import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.fit.ItemFitter;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.ProbabilisticClassifier;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;

/**
 * @author tyler
 */
public class ItemClassifier
        extends ProbabilisticClassifier<Vector, ItemClassifier, ItemClassificationModel>
        implements Cloneable
{
    private static final long serialVersionUID = 0x7cc313e747f68becL;

    private final ItemClassifierSettings _settings;
    private final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> _startingParams;
    private String _uid;

    public ItemClassifier(final ItemClassifierSettings settings_)
    {
        this(settings_, null);
    }

    public ItemClassifier(final ItemClassifierSettings settings_, final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> startingParams_)
    {
        if (null == settings_)
        {
            throw new NullPointerException("Settings cannot be null.");
        }

        _settings = settings_;
        _startingParams = startingParams_;
    }

    @Override
    public ItemClassifier copy(ParamMap paramMap_)
    {
        return this.defaultCopy(paramMap_);
    }

    private ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> generateFitter(final Dataset<?> data_)
    {
        //This is pretty filthy, but it will get the job done. Though, only locally.
        final String featureCol = this.getFeaturesCol();
        final String labelCol = this.getLabelCol();

        final ItemStatusGrid<SimpleStatus, SimpleRegressor> data = new SparkGridAdapter(data_, labelCol, featureCol,
                this._settings.getRegressors(), this._settings.getFromStatus(), _settings.getIntercept());

        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = new ItemFitter<>(_settings.getFactory(),
                _settings.getIntercept(), _settings.getFromStatus(), data);

        return fitter;
    }


    public ItemClassificationModel runAnnealing(final Dataset<?> data_, ItemClassificationModel prevModel_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);

        try
        {
            fitter.pushParameters("PrevModel", prevModel_.getParams());

            //Now run full scale annealing.
            fitter.runAnnealingByEntry(_settings.getCurveRegressors(), false);

            final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> params = fitter.getBestParameters();
            final ItemClassificationModel classificationModel = new ItemClassificationModel(params, _settings.getRegressors());

            return classificationModel;
        }
        catch (final ConvergenceException e)
        {
            //TODO: Fix this, this exception is actually not possible here.
            throw new RuntimeException(e);
        }
    }

    public ItemClassificationModel retrainModel(final Dataset<?> data_, ItemClassificationModel prevModel_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);

        try
        {
            fitter.pushParameters("PrevModel", prevModel_.getParams());
        }
        catch (final ConvergenceException e)
        {
            //TODO: Fix this, this exception is actually not possible here.
            throw new RuntimeException(e);
        }


        final int maxParams = _settings.getMaxParamCount();

        try
        {
            final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
            final int remainingParams = maxParams - usedParams;

            //Starts out with vacuous intercepts, correct those first.
            fitter.fitCoefficients();

            if (remainingParams > 1)
            {
                //Now add the flags, recalibrate beta values.
                fitter.addCoefficients(_settings.getNonCurveRegressors());
            }

            final int curveAvailable = maxParams - fitter.getBestParameters().getEffectiveParamCount();

            if (curveAvailable > 3)
            {
                //Now expand the model by adding curves.
                fitter.expandModel(_settings.getCurveRegressors(), curveAvailable);
            }

            //Trim anything rendered irrelevant by later passes.
            fitter.trim(true);
        }
        catch (final ConvergenceException e)
        {
            throw new RuntimeException(e);
        }

        final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> params = fitter.getBestParameters();
        final ItemClassificationModel classificationModel = new ItemClassificationModel(params, _settings.getRegressors());

        return classificationModel;
    }


    @Override
    public ItemClassificationModel train(final Dataset<?> data_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);

        if (null != _startingParams)
        {
            try
            {
                fitter.pushParameters("InitialParams", _startingParams);
            }
            catch (final ConvergenceException e)
            {
                //TODO: Fix this, this exception is actually not possible here.
                throw new RuntimeException(e);
            }
        }

        final int maxParams = _settings.getMaxParamCount();

        try
        {
            final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
            final int remainingParams = maxParams - usedParams;

            //Starts out with vacuous intercepts, correct those first.
            fitter.fitCoefficients();

            if (remainingParams > 1)
            {
                //Now add the flags, recalibrate beta values.
                fitter.addCoefficients(_settings.getNonCurveRegressors());
            }

            final int curveAvailable = maxParams - fitter.getBestParameters().getEffectiveParamCount();

            if (curveAvailable > 3)
            {
                //Now expand the model by adding curves.
                fitter.expandModel(_settings.getCurveRegressors(), curveAvailable);
            }

            //Trim anything rendered irrelevant by later passes.
            fitter.trim(true);
        }
        catch (final ConvergenceException e)
        {
            throw new RuntimeException(e);
        }

        final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> params = fitter.getBestParameters();
        final ItemClassificationModel classificationModel = new ItemClassificationModel(params, _settings.getRegressors());
        return classificationModel;
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

}
