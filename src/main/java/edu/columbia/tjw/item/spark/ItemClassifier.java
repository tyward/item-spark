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

import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.fit.ItemFitter;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.util.random.RandomTool;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.spark.ml.classification.ProbabilisticClassifier;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public class ItemClassifier<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> extends ProbabilisticClassifier<Vector, ItemClassifier<S, R, T>, ItemClassificationModel<S, R, T>> {

    private static final long serialVersionUID = 0x7cc313e747f68becL;

    private final ItemSettings _settings;
    private final S _fromStatus;
    private final R _intercept;
    private final ItemCurveFactory<R, T> _factory;
    private final String _uid;
    private final int _maxParamCount;

    public ItemClassifier(final ItemSettings settings_, final ItemCurveFactory<R, T> factory_, final R intercept_, final S status_, final int maxParamCount_) {
        if (null == settings_) {
            _settings = new ItemSettings();
        } else {
            _settings = settings_;
        }

        _uid = RandomTool.randomString(64);

        _factory = factory_;
        _fromStatus = status_;
        _intercept = intercept_;
        _maxParamCount = maxParamCount_;
    }

    @Override
    public ItemClassifier<S, R, T> copy(ParamMap paramMap_) {
        final ItemSettings settings = paramMap_.apply(ItemSettingsParam.singleton());
        return new ItemClassifier<>(settings, _factory, _intercept, _fromStatus, _maxParamCount);
    }

    @Override
    public ItemClassificationModel<S, R, T> train(final Dataset<?> data_) {
        //This is pretty filthy, but it will get the job done. Though, only locally. 
        final String featureCol = this.getFeaturesCol();
        final String labelCol = this.getLabelCol();

        final Iterator<Row> rowForm = (Iterator<Row>) data_.toLocalIterator();
        final List<double[]> rows = new ArrayList<>();
        final List<Integer> toStatus = new ArrayList<>();

        while (rowForm.hasNext()) {
            final Row next = rowForm.next();
            final Vector vec = (Vector) next.get(next.fieldIndex(featureCol));
            final Number label = (Number) next.getAs(next.fieldIndex(labelCol));
            rows.add(vec.toArray());
            toStatus.add(label.intValue());

        }

        //Generate the data set...
        final ItemStatusGrid<S, R> data = null;

        final ItemFitter<S, R, T> fitter = new ItemFitter<>(_factory, _intercept, _fromStatus, data);

        try {
            for (int i = 0; i < 3; i++) {
                final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
                final int remainingParams = _maxParamCount - usedParams;

                if (remainingParams < 1) {
                    //Not enough params to do anything, break out.
                    break;
                }

                //Starts out with vacuous intercepts, correct those first. 
                fitter.fitCoefficients(null);

                //Now add the flags, recalibrate beta values.
                fitter.addCoefficients(null, null);

                final int curveAvailable = _maxParamCount - fitter.getBestParameters().getEffectiveParamCount();

                if (curveAvailable > 3) {
                    //Now expand the model by adding curves.
                    fitter.expandModel(null, null, curveAvailable);
                } else {
                    break;
                }
                
                //Relaxation based calibration of current curves.
                fitter.calibrateCurves();

                //Refit coefficients.
                fitter.fitCoefficients(null);

                //Trim anything rendered irrelevant by later passes.
                fitter.trim(true);

                //Now run full scale annealing.
                fitter.runAnnealingByEntry(null, false);
            }

        } catch (final ConvergenceException e) {
            throw new RuntimeException(e);
        }

        final ItemParameters<S, R, T> params = fitter.getBestParameters();

        final ItemClassificationModel<S, R, T> classificationModel = new ItemClassificationModel<>(params);

        return classificationModel;
    }

    @Override
    public String uid() {
        return _uid;
    }

}
