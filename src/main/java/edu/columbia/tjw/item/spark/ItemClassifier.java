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
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemStatus;
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
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public class ItemClassifier<S extends ItemStatus<S>, R extends ItemRegressor<R>>
        extends ProbabilisticClassifier<Vector, ItemClassifier<S, R>, ItemClassificationModel<S, R>>
        implements Cloneable
{

    private static final long serialVersionUID = 0x7cc313e747f68becL;

    private ItemClassifierSettings<S, R> _settings;
    private String _uid;

    public ItemClassifier(final ItemClassifierSettings<S, R> settings_)
    {
        if (null == settings_)
        {
            throw new NullPointerException("Settings cannot be null.");
        }

        _settings = settings_;
    }

    private void updateSettings(ItemClassifierSettings<S, R> settings_)
    {
        if (null == settings_)
        {
            throw new NullPointerException("Settings cannot be null.");
        }

        _settings = settings_;
    }

    @Override
    public ItemClassifier<S, R> copy(ParamMap paramMap_)
    {
        return this.defaultCopy(paramMap_);
    }

    @Override
    public ItemClassificationModel<S, R> train(final Dataset<?> data_)
    {
        //This is pretty filthy, but it will get the job done. Though, only locally. 
        final String featureCol = this.getFeaturesCol();
        final String labelCol = this.getLabelCol();

        final ItemStatusGrid<S, R> data = new SparkGridAdapter<>(data_, labelCol, featureCol,
                this._settings.getRegressors(), this._settings.getFromStatus(), _settings.getIntercept());

        final ItemFitter<S, R, StandardCurveType> fitter = new ItemFitter<>(_settings.getFactory(),
                _settings.getIntercept(), _settings.getFromStatus(), data);

        final int maxParams = _settings.getMaxParamCount();

        try
        {
            for (int i = 0; i < 3; i++)
            {
                final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
                final int remainingParams = maxParams - usedParams;

                if (remainingParams < 1)
                {
                    //Not enough params to do anything, break out.
                    break;
                }

                //Starts out with vacuous intercepts, correct those first. 
                fitter.fitCoefficients(null);

                //Now add the flags, recalibrate beta values.
                fitter.addCoefficients(null, _settings.getNonCurveRegressors());

                final int curveAvailable = maxParams - fitter.getBestParameters().getEffectiveParamCount();

                if (curveAvailable > 3)
                {
                    //Now expand the model by adding curves.
                    fitter.expandModel(_settings.getCurveRegressors(), null, curveAvailable);
                }
                else
                {
                    break;
                }

                //Relaxation based calibration of current curves.
                fitter.calibrateCurves();

                //Refit coefficients.
                fitter.fitCoefficients(null);

                //Trim anything rendered irrelevant by later passes.
                fitter.trim(true);

                //Now run full scale annealing.
                fitter.runAnnealingByEntry(_settings.getCurveRegressors(), false);
            }

        }
        catch (final ConvergenceException e)
        {
            throw new RuntimeException(e);
        }

        final ItemParameters<S, R, StandardCurveType> params = fitter.getBestParameters();

        final ItemClassificationModel<S, R> classificationModel = new ItemClassificationModel<>(params, _settings.getRegressors());

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
