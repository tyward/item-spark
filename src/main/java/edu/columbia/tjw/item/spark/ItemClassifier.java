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
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.base.raw.RawFittingGrid;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.fit.FitResult;
import edu.columbia.tjw.item.fit.GradientResult;
import edu.columbia.tjw.item.fit.ItemFitter;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.util.EnumFamily;
import edu.columbia.tjw.item.util.random.RandomTool;
import org.apache.spark.ml.classification.ProbabilisticClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;

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

    public ItemClassifier(final ItemClassifierSettings settings_,
                          final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> startingParams_)
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

    public ItemClassifierSettings getSettings()
    {
        return _settings;
    }

    public RawFittingGrid<SimpleStatus, SimpleRegressor> generateMaterializedGrid(final Dataset<?> data_)
    {
        return new RawFittingGrid<>(generateFitter(data_).getGrid());
    }

    private ItemStatusGrid<SimpleStatus, SimpleRegressor> generateGrid(final Dataset<?> data_)
    {
        //This is pretty filthy, but it will get the job done. Though, only locally.
        final String featureCol = this.getFeaturesCol();
        final String labelCol = this.getLabelCol();

        final ItemStatusGrid<SimpleStatus, SimpleRegressor> data = new SparkGridAdapter(data_, labelCol, featureCol,
                this._settings.getRegressors(), this._settings.getFromStatus(), _settings.getRegressorFamily());

        return data;
    }


    private ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> generateFitter(final Dataset<?> data_)
    {
        final ItemStatusGrid<SimpleStatus, SimpleRegressor> data = generateGrid(data_);

        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = new ItemFitter<>(
                _settings.getFactory(),
                _settings.getRegressorFamily(), _settings.getFromStatus(), data, _settings.getSettings());

        return fitter;
    }

    public static Dataset<Row> prepareData(final Dataset<?> data_, final ItemClassifierSettings settings_,
                                           final String featuresColumn_)
    {
        final List<SimpleRegressor> regs = settings_.getRegressors();
        final String[] regNames = new String[regs.size()];
        int pointer = 0;

        for (final SimpleRegressor reg : regs)
        {
            regNames[pointer++] = reg.name();
        }

        final VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(regNames);
        assembler.setOutputCol(featuresColumn_);

        final Dataset<Row> withFeatures = assembler.transform(data_);
        return withFeatures;
    }


    public static ItemClassifierSettings prepareSettings(final Dataset<?> data_, final String toStatusColumn_,
                                                         final List<String> featureList,
                                                         final Set<String> curveRegressors_, final int maxParamCount_)
    {
        return prepareSettings(data_, toStatusColumn_, featureList, curveRegressors_, maxParamCount_,
                new ItemSettings());
    }

    public static ItemClassifierSettings prepareSettings(final Dataset<?> data_, final String toStatusColumn_,
                                                         final List<String> featureList,
                                                         final Set<String> curveRegressors_, final int maxParamCount_,
                                                         final ItemSettings settings_)
    {
        final Iterator<?> iter = data_.select(toStatusColumn_).distinct().toLocalIterator();
        final SortedMap<Integer, Integer> statSet = new TreeMap<>();

        while (iter.hasNext())
        {
            final Row nextRow = (Row) iter.next();
            final Object nextObj = nextRow.get(0);

            if (null == nextObj)
            {
                continue;
            }

            final Integer statIndex = ((Number) nextObj).intValue();
            statSet.putIfAbsent(statIndex, 0);
            final Integer prevVal = statSet.get(statIndex);
            statSet.put(statIndex, prevVal + 1);
        }

        // Base status will be the most common to-status. This tends to make the resulting parameters more
        // understandable.
        int baseStatus = -1;
        int baseCount = 0;

        for (final Map.Entry<Integer, Integer> entry : statSet.entrySet())
        {
            final int entryCount = entry.getValue();

            if (entryCount > baseCount)
            {
                baseStatus = entry.getKey();
                baseCount = entryCount;
            }
        }


        final List<String> statList = new ArrayList<>();
        int baseIndex = -1;

        for (final Integer next : statSet.keySet())
        {
            statList.add(next.toString());

            if (next.intValue() == baseStatus)
            {
                baseIndex = statList.size();
            }
        }

        final EnumFamily<SimpleStatus> statFamily = SimpleStatus.generateFamily(statList);
        final SimpleStatus baseStatusMember = statFamily.getFromOrdinal(baseIndex);

//        final List<String> regList = new ArrayList<>();
//        //regList.add(INTERCEPT_NAME);
//        regList.addAll(featureList);

        final Set<String> distinctSet = new HashSet<>(featureList);

        if (distinctSet.size() != featureList.size())
        {
            throw new RuntimeException("Non distinct features: " + featureList.size());
        }
        if (!distinctSet.containsAll(curveRegressors_))
        {
            throw new RuntimeException("All curve regressors must also be in the feature list.");
        }

        final ItemClassifierSettings settings = new ItemClassifierSettings(settings_,
                baseStatusMember, maxParamCount_, featureList, curveRegressors_);

        return settings;
    }


    public GradientResult computeGradients(final Dataset<?> data_, ItemClassificationModel model_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);
        return fitter.getCalculator().computeGradients(model_.getParams());
    }

    public FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> computeFitResult(final Dataset<?> data_,
                                                                                        ItemClassificationModel model_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);
        return fitter.getCalculator().computeFitResult(model_.getParams(), null);
    }

    public ItemClassificationModel runAnnealing(final Dataset<?> data_, ItemClassificationModel prevModel_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);

        try
        {
            fitter.pushParameters("PrevModel", prevModel_.getParams());

            //Now run full scale annealing.
            fitter.runAnnealingByEntry(_settings.getCurveRegressors(), true);

            final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult = fitter.getChain()
                    .getLatestResults();
            final ItemClassificationModel classificationModel = new ItemClassificationModel(fitResult, _settings);

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
        fitter.pushParameters("PrevModel", prevModel_.getParams());

        final int maxParams = _settings.getMaxParamCount();

        try
        {
            final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
            final int remainingParams = maxParams - usedParams;

            fitter.fitModel(_settings.getNonCurveRegressors(), _settings.getCurveRegressors(), remainingParams, false);
        }
        catch (final ConvergenceException e)
        {
            throw new RuntimeException(e);
        }

        final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult =
                fitter.getChain().getLatestResults();
        final ItemClassificationModel classificationModel = new ItemClassificationModel(fitResult, _settings);

        return classificationModel;
    }

//    public ItemClassificationModel generateModel(final Dataset<?> data_, final ItemParameters<SimpleStatus,
//            SimpleRegressor,
//            StandardCurveType> params_) {
//        final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult =
//                fitter.getChain().getLatestResults();
//    }

    @Override
    public ItemClassificationModel train(final Dataset<?> data_)
    {
        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = generateFitter(data_);

        if (null != _startingParams)
        {
            fitter.pushParameters("InitialParams", _startingParams);
        }

        final int maxParams = _settings.getMaxParamCount();

        try
        {
            final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
            final int remainingParams = maxParams - usedParams;

            fitter.fitModel(_settings.getNonCurveRegressors(), _settings.getCurveRegressors(), remainingParams, false);
        }
        catch (final ConvergenceException e)
        {
            throw new RuntimeException(e);
        }

        final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult =
                fitter.getChain().getLatestResults();
        final ItemClassificationModel classificationModel = new ItemClassificationModel(fitResult,
                _settings);
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
