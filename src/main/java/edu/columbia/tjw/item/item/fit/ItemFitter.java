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
package edu.columbia.tjw.item.fit;

import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.ParamFilter;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.data.RandomizedStatusGrid;
import edu.columbia.tjw.item.fit.EntropyCalculator.EntropyAnalysis;
import edu.columbia.tjw.item.fit.FittingProgressChain.ParamProgressFrame;
import edu.columbia.tjw.item.fit.curve.CurveFitter;
import edu.columbia.tjw.item.fit.param.ParamFitResult;
import edu.columbia.tjw.item.fit.param.ParamFitter;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.util.EnumFamily;
import edu.columbia.tjw.item.util.LogUtil;
import java.util.Collection;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.logging.Logger;

/**
 *
 * A class designed to expand the model by adding curves.
 *
 * In addition, it may be used to fit only coefficients if needed.
 *
 *
 * @author tyler
 * @param <S> The status type for this model
 * @param <R> The regressor type for this model
 * @param <T> The curve type for this model
 */
public final class ItemFitter<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private static final Logger LOG = LogUtil.getLogger(ItemFitter.class);

    private final ItemCurveFactory<R, T> _factory;
    private final ItemSettings _settings;
    private final R _intercept;
    private final EnumFamily<R> _family;
    private final S _status;
    private final ItemStatusGrid<S, R> _grid;
    private final EntropyCalculator<S, R, T> _calc;

    private final ParamFitter<S, R, T> _fitter;
    final CurveFitter<S, R, T> _curveFitter;

    private final FittingProgressChain<S, R, T> _chain;

    public ItemFitter(final ItemCurveFactory<R, T> factory_, final R intercept_, final S status_, final ItemStatusGrid<S, R> grid_)
    {
        this(factory_, intercept_, status_, grid_, new ItemSettings());
    }

    public ItemFitter(final ItemCurveFactory<R, T> factory_, final R intercept_, final S status_, final ItemStatusGrid<S, R> grid_, ItemSettings settings_)
    {
        if (null == factory_)
        {
            throw new NullPointerException("Factory cannot be null.");
        }
        if (null == intercept_)
        {
            throw new NullPointerException("Intercept cannot be null.");
        }
        if (null == settings_)
        {
            throw new NullPointerException("Settings cannot be null.");
        }
        if (null == status_)
        {
            throw new NullPointerException("Status cannot be null.");
        }
        if (null == grid_)
        {
            throw new NullPointerException("Grid cannot be null.");
        }

        _factory = factory_;
        _settings = settings_;
        _intercept = intercept_;
        _family = intercept_.getFamily();
        _status = status_;
        _grid = randomizeGrid(grid_, _settings);
        _calc = new EntropyCalculator<>(_grid, _status, _settings);

        final ItemParameters<S, R, T> starting = new ItemParameters<>(status_, _intercept);
        final double logLikelihood = this.computeLogLikelihood(starting);
        _chain = new FittingProgressChain<>("Primary", starting, logLikelihood, _grid.size(), _calc, _settings.getDoValidate());

        _fitter = new ParamFitter<>(_calc, _settings, null);
        _curveFitter = new CurveFitter<>(_factory, _settings, _grid, _calc);
    }

    public S getStatus()
    {
        return _status;
    }

    public FittingProgressChain<S, R, T> getChain()
    {
        return _chain;
    }

    public EntropyCalculator<S, R, T> getCalculator()
    {
        return _calc;
    }

    public double getBestLogLikelihood()
    {
        return _chain.getLogLikelihood();
    }

    public ItemParameters<S, R, T> getBestParameters()
    {
        return _chain.getBestParameters();
    }

    /**
     * This function will wrap the provided grid factory. The goal here is to
     * handle the randomization needed for accurate calculation, and also to
     * cache some data for efficiency.
     *
     * It is strongly recommended that all grid factories are wrapped before
     * use.
     *
     * N.B: The wrapped grid may cache, so if the underlying regressors are
     * changed, the resulting factory should be wrapped again.
     *
     * Also, keep in mind that only relevant rows will be retained, particularly
     * those with the correct from state, and for which the next status is
     * known.
     *
     * @param grid_ The grid to randomize
     * @param settings_
     * @return A randomized version of grid_
     */
    public final ItemStatusGrid<S, R> randomizeGrid(final ItemStatusGrid<S, R> grid_,
            final ItemSettings settings_)
    {
        if (grid_ instanceof RandomizedStatusGrid)
        {
            return grid_;
        }

        final ItemStatusGrid<S, R> wrapped = new RandomizedStatusGrid<>(grid_, settings_, grid_.getRegressorFamily(), _status);
        return wrapped;
    }

    public ItemStatusGrid<S, R> getGrid()
    {
        return _grid;
    }

    public ParamFitResult<S, R, T> pushParameters(final String label_, ItemParameters<S, R, T> params_) throws ConvergenceException
    {
        _chain.forcePushResults("ForcePush[" + label_ + "]", params_);
        return _chain.getLatestResults();
    }

    /**
     * Add a group of coefficients to the model, then refit all coefficients.
     *
     * @param filters_ Any filters that should be used to limit the allowed
     * coefficients, else null
     * @param coefficients_ The set of coefficients to fit.
     * @return A model fit with all the additional allowed coefficients.
     * @throws ConvergenceException If no progress could be made
     */
    public ParamFitResult<S, R, T> addCoefficients(final Collection<ParamFilter<S, R, T>> filters_,
            final Collection<R> coefficients_) throws ConvergenceException
    {
        ItemParameters<S, R, T> params = _chain.getBestParameters();

        final SortedSet<R> flagSet = new TreeSet<>();

        for (int i = 0; i < params.getEntryCount(); i++)
        {
            if (params.getEntryDepth(i) != 1)
            {
                continue;
            }
            if (params.getEntryCurve(i, 0) != null)
            {
                continue;
            }

            flagSet.add(params.getEntryRegressor(i, 0));
        }

        final int startingSize = params.getEntryCount();

        for (final R field : coefficients_)
        {
            if (flagSet.contains(field))
            {
                continue;
            }

            params = params.addBeta(field);
        }

        if (params.getEntryCount() != startingSize)
        {
            _chain.pushVacuousResults("VacuousAddCoefficients", params);
        }

        innerFitCoefficients(_chain, filters_);
        return _chain.getLatestResults();
    }

    /**
     *
     * Optimize the coefficients.
     *
     * @param filters_ Filters describing any coefficients that should not be
     * adjusted
     * @return A model with newly optimized coefficients.
     * @throws ConvergenceException If no progress could be made
     */
    public ParamFitResult<S, R, T> fitCoefficients(final Collection<ParamFilter<S, R, T>> filters_) throws ConvergenceException
    {
        innerFitCoefficients(_chain, filters_);
        return _chain.getLatestResults();
    }

    private void innerFitCoefficients(final FittingProgressChain<S, R, T> chain_, final Collection<ParamFilter<S, R, T>> filters_) throws ConvergenceException
    {
        //final ParamFitter<S, R, T> fitter = new ParamFitter<>(chain_.getBestParameters(), _grid, _settings, filters_);
        _fitter.fit(chain_);
        //chain_.pushResults("FitCoefficients", fitResult);
    }

    private void doSingleAnnealingOperation(final Set<R> curveFields_, final ItemParameters<S, R, T> base_, final ItemParameters<S, R, T> reduced_,
            final FittingProgressChain<S, R, T> subChain_, final boolean exhaustiveCalibrate_)
    {
        final int paramCount = base_.getEffectiveParamCount();

        subChain_.forcePushResults("ReducedFrame", reduced_);

        if (exhaustiveCalibrate_)
        {
            try
            {
                this.innerFitCoefficients(subChain_, null);
                _curveFitter.calibrateCurves(0.0, true, subChain_);
            }
            catch (final ConvergenceException e)
            {
                e.printStackTrace();
            }
        }

        final int reducedCount = reduced_.getEffectiveParamCount();
        final int reduction = paramCount - reducedCount;

        if (reduction <= 0)
        {
            return;
        }

        final ParamFitResult<S, R, T> rebuilt = expandModel(subChain_, curveFields_, null, reduction);
        final boolean better = _chain.pushResults("AnnealingExpansion", subChain_.getBestParameters(), subChain_.getLogLikelihood());

        if (better)
        {
            LOG.info("Annealing improved model: " + rebuilt.getStartingLL() + " -> " + rebuilt.getEndingLL() + " (" + rebuilt.getAic() + ")");
        }
        else
        {
            LOG.info("Annealing did not improve model, keeping old model");
        }
    }

    public ParamFitResult<S, R, T> trim(final boolean exhaustiveCalibration_)
    {
        return this.trim(-_settings.getAicCutoff(), exhaustiveCalibration_);
    }

    public ParamFitResult<S, R, T> trim(final double aicCutoff_, final boolean exhaustiveCalibration_)
    {
        for (int i = 0; i < _chain.getBestParameters().getEntryCount(); i++)
        {
            final FittingProgressChain<S, R, T> subChain = new FittingProgressChain<>("AnnealingSubChain", _chain);
            final ItemParameters<S, R, T> base = subChain.getBestParameters();

            if (i == base.getInterceptIndex())
            {
                continue;
            }

            final ItemParameters<S, R, T> reduced = base.dropIndex(i);

            subChain.forcePushResults("DropEntry", reduced);

            if (exhaustiveCalibration_)
            {
                try
                {
                    _fitter.fit(subChain);
                }
                catch (final ConvergenceException e)
                {
                    LOG.info("Convergence Exception: " + e);
                }
            }

            final double aic = subChain.getLatestFrame().getAicDiff();

            //This entry is not good enough, so remove it.
            if (aic < aicCutoff_)
            {
                LOG.info("Trimming an entry[" + i + "]");
                _chain.forcePushResults("Trim", reduced);

                //Just step back, this entry has been removed, other entries slid up.
                i--;
            }
        }

        return _chain.getLatestResults();
    }

    public ParamFitResult<S, R, T> runAnnealingByEntry(final Set<R> curveFields_, final boolean exhaustiveCalibration_) throws ConvergenceException
    {
        int offset = 0;

        for (int i = 0; i < _chain.getBestParameters().getEntryCount(); i++)
        {
            final FittingProgressChain<S, R, T> subChain = new FittingProgressChain<>("AnnealingSubChain[" + i + "]", _chain);
            final ItemParameters<S, R, T> base = subChain.getBestParameters();
            final int index = i - offset;

            if (index == base.getInterceptIndex())
            {
                continue;
            }
            if (base.getEntryStatusRestrict(index) == null)
            {
                //Annealing is only applied to curve entries.
                continue;
            }

            final ItemParameters<S, R, T> reduced = base.dropIndex(index);
            doSingleAnnealingOperation(curveFields_, base, reduced, subChain, exhaustiveCalibration_);

            final ParamFitResult<S, R, T> results = subChain.getConsolidatedResults();
            final double aic = results.getAic();

            LOG.info("----->Completed Annealing Step[" + i + "]: " + aic);

            if (aic < _settings.getAicCutoff())
            {
                //Just step back, this entry has been removed, other entries slid up.
                offset++;
                //_chain.pushResults(subChain.getName(), subChain.getConsolidatedResults());
            }
        }

        return _chain.getLatestResults();
    }

    public ParamFitResult<S, R, T> runAnnealingPass(final Set<R> curveFields_, final boolean exhaustiveCalibration_) throws ConvergenceException
    {
        for (final R regressor : curveFields_)
        {
            final FittingProgressChain<S, R, T> subChain = new FittingProgressChain<>("AnnealingSubChain[" + regressor.name() + "]", _chain);
            final ItemParameters<S, R, T> base = subChain.getBestParameters();
            final ItemParameters<S, R, T> reduced = base.dropRegressor(regressor);

            LOG.info("Annealing attempting to drop params from " + regressor);
            doSingleAnnealingOperation(curveFields_, base, reduced, subChain, exhaustiveCalibration_);
            LOG.info("---->Finished rebuild after dropping regressor: " + regressor);
        }

        return _chain.getLatestResults();
    }

    public double computeLogLikelihood(final ItemParameters<S, R, T> params_)
    {
        final EntropyAnalysis ea = _calc.computeEntropy(params_);
        final double entropy = ea.getEntropy();
        return entropy;
    }

    public ParamFitResult<S, R, T> generateFlagInteractions(final boolean exhaustive_)
    {
        return generateFlagInteractions(_chain.getBestParameters().getEntryCount(), exhaustive_);
    }

    private ParamFitResult<S, R, T> generateFlagInteractions(final int entryNumber_, final boolean exhaustive_)
    {
        //N.B: This loop can keep expanding as the params grows very large, if we are very successful.
        // Just make sure to cap it out at the entryNumber_
        for (int i = 0; i < Math.min(_chain.getBestParameters().getEntryCount(), entryNumber_); i++)
        {
            final ItemParameters<S, R, T> params = _chain.getBestParameters();

            if (i == params.getInterceptIndex())
            {
                continue;
            }
            if (params.getEntryStatusRestrict(i) != null)
            {
                //This would be a curve, but curves already get interactions when generated or annealed.
                continue;
            }

            _curveFitter.generateInteractions(_chain, params, params.getEntryCurveParams(i, true),
                    params.getEntryStatusRestrict(i), 0.0, _chain.getLogLikelihood(), exhaustive_);
        }

        return _chain.getLatestResults();
    }

    /**
     * Add some new curves to this model.
     *
     * This function is the heart of the ITEM system, and uses most of the
     * computational resources.
     *
     * @param curveFields_ The regressors on which to draw curves
     * @param filters_ Filters describing any curves that should not be drawn or
     * optimized
     * @param paramCount_ The total number of additional params that will be
     * allowed.
     * @return A new model with additional curves added, and all coefficients
     * optimized.
     */
    public ParamFitResult<S, R, T> expandModel(final Set<R> curveFields_,
            final Collection<ParamFilter<S, R, T>> filters_, final int paramCount_)
    {
        if (paramCount_ < 1)
        {
            throw new IllegalArgumentException("Param count must be positive.");
        }

        expandModel(_chain, curveFields_, filters_, paramCount_);
        return _chain.getLatestResults();
    }

    public ParamFitResult<S, R, T> calibrateCurves()
    {
        final FittingProgressChain<S, R, T> subChain = new FittingProgressChain<>("CalibrationChain", _chain);

        //First, try to calibrate any existing curves to improve the fit. 
        _curveFitter.calibrateCurves(0.0, true, subChain);

        final ParamFitResult<S, R, T> results = subChain.getConsolidatedResults();

        this._chain.pushResults("ExhaustiveCalibration", results);

        return results;
    }

    private ParamFitResult<S, R, T> expandModel(final FittingProgressChain<S, R, T> chain_, final Set<R> curveFields_,
            final Collection<ParamFilter<S, R, T>> filters_, final int paramCount_)
    {
        final long start = System.currentTimeMillis();
        //final FittingProgressChain<S, R, T> subChain = new FittingProgressChain<>("ModelExpansionChain", chain_);

        final int statingParamCount = chain_.getBestParameters().getEffectiveParamCount();
        final int paramCountLimit = statingParamCount + paramCount_;

        //As a bare minimum, each expansion will consume at least one param, we'll break out before this most likely.
        for (int i = 0; i < paramCount_; i++)
        {
            try
            {
                innerFitCoefficients(chain_, filters_);
            }
            catch (final ConvergenceException e)
            {
                LOG.warning("Unable to improve results in coefficient fit, moving on.");
            }

            if (i > 0)
            {
                final ParamProgressFrame frame = chain_.getLatestFrame();
                final double improvement = frame.getStartingPoint().getCurrentLogLikelihood() - frame.getCurrentLogLikelihood();

                //First, try to calibrate any existing curves to improve the fit. 
                final boolean isBetter = _curveFitter.calibrateCurves(improvement, false, chain_);

                if (!isBetter)
                {
                    LOG.info("Curve calibration unable to improve results.");
                }
            }

            try
            {
                //Now, try to add a new curve. 
                final boolean expansionBetter = _curveFitter.generateCurve(chain_, curveFields_, filters_);

                if (!expansionBetter)
                {
                    LOG.info("Curve expansion unable to improve results, breaking out.");
                    break;
                }

                if (chain_.getBestParameters().getEffectiveParamCount() >= paramCountLimit)
                {
                    LOG.info("Param count limit reached, breaking out.");
                    break;
                }

            }
            catch (final ConvergenceException e)
            {
                LOG.info("Unable to make progress, breaking out.");
                break;
            }
            finally
            {
                LOG.info("Completed one round of curve drawing, moving on.");
                LOG.info("Time marker: " + (System.currentTimeMillis() - start));
                LOG.info("Heap used: " + Runtime.getRuntime().totalMemory() / (1024 * 1024));
            }
        }

        try
        {
            innerFitCoefficients(chain_, filters_);
        }
        catch (final ConvergenceException e)
        {
            LOG.info("Unable to improve results in coefficient fit, moving on.");
        }

        return chain_.getConsolidatedResults();
    }

}
