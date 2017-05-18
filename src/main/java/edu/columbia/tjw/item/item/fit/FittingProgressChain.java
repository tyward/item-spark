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

import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemParameters;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.fit.EntropyCalculator.EntropyAnalysis;
import edu.columbia.tjw.item.fit.curve.CurveFitResult;
import edu.columbia.tjw.item.fit.param.ParamFitResult;
import edu.columbia.tjw.item.util.LogUtil;
import edu.columbia.tjw.item.util.MathFunctions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 * @param <T>
 */
public final class FittingProgressChain<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>>
{
    private static final Logger LOG = LogUtil.getLogger(FittingProgressChain.class);

    private final String _chainName;
    private final List<ParamProgressFrame<S, R, T>> _frameList;
    private final List<ParamProgressFrame<S, R, T>> _frameListReadOnly;
    private final int _rowCount;
    private final EntropyCalculator<S, R, T> _calc;
    private final boolean _validate;

    /**
     * Start a new chain from the latest entry of the current chain.
     *
     * @param chainName_
     * @param baseChain_
     */
    public FittingProgressChain(final String chainName_, final FittingProgressChain<S, R, T> baseChain_)
    {
        this(chainName_, baseChain_.getBestParameters(), baseChain_.getLogLikelihood(), baseChain_.getRowCount(), baseChain_._calc, baseChain_.isValidate());
    }

    public FittingProgressChain(final String chainName_, final ItemParameters<S, R, T> fitResult_, final double startingLL_, final int rowCount_, final EntropyCalculator<S, R, T> calc_, final boolean validating_)
    {
        if (rowCount_ <= 0)
        {
            throw new IllegalArgumentException("Data set cannot be empty.");
        }

        this.validate(fitResult_, startingLL_);

        _chainName = chainName_;
        _rowCount = rowCount_;
        final ParamProgressFrame<S, R, T> frame = new ParamProgressFrame<>("Initial", fitResult_, startingLL_, null, _rowCount);

        _frameList = new ArrayList<>();
        _frameList.add(frame);
        _frameListReadOnly = Collections.unmodifiableList(_frameList);
        _calc = calc_;
        _validate = validating_;
    }

    public String getName()
    {
        return _chainName;
    }

    public boolean pushResults(final String frameName_, final CurveFitResult<S, R, T> curveResult_)
    {
        final double currLL = this.getLogLikelihood();
        final double incomingStartLL = curveResult_.getStartingLogLikelihood();

        final double incomingEntropy = curveResult_.getLogLikelihood();
        final ItemParameters<S, R, T> incomingParams = curveResult_.getModelParams();

        //This should always match, the curve result was built on top of this chain, right? 
        final int compare = MathFunctions.doubleCompareRounded(currLL, incomingStartLL);

        if (compare != 0)
        {
            final EntropyAnalysis ea = _calc.computeEntropy(curveResult_.getStartingParams());

            LOG.info("Unexpected incoming Log Likelihood: " + currLL + " != "
                    + incomingStartLL + " (" + ea.getEntropy() + ")");
        }

        return this.pushResults(frameName_, incomingParams, incomingEntropy);
    }

    private synchronized void validate(final ItemParameters<S, R, T> fitResult_, final double entropy_)
    {
        if (this.isValidate())
        {

            //Since the claim is that the LL improved, let's see if that's true...
            final EntropyAnalysis ea = _calc.computeEntropy(fitResult_);
            final double entropy = ea.getEntropy();

            //LOG.info("Params: " + fitResult_.hashCode() + " -> " + entropy);
            //LOG.info("Chain: " + this.toString());
            final int compare = MathFunctions.doubleCompareRounded(entropy, entropy_);

            if (compare != 0)
            {
                throw new IllegalStateException("Found entropy mismatch: " + entropy + " != " + entropy_);
            }
        }
    }

    public boolean pushResults(final String frameName_, final ParamFitResult<S, R, T> fitResult_)
    {
        final double currLL = this.getLogLikelihood();
        final double incomingStartLL = fitResult_.getStartingLL();

        final int compare = MathFunctions.doubleCompareRounded(currLL, incomingStartLL);

        if (compare != 0)
        {
            final EntropyAnalysis ea = _calc.computeEntropy(fitResult_.getStartingParams());

            LOG.info("Unexpected incoming Log Likelihood: " + currLL + " != "
                    + incomingStartLL + " (" + ea.getEntropy() + ")");
        }

        return this.pushResults(frameName_, fitResult_.getEndingParams(), fitResult_.getEndingLL());
    }

    /**
     * Forces the given results onto the stack, regardless of quality...
     *
     * @param frameName_
     * @param fitResult_
     */
    public void forcePushResults(final String frameName_, final ItemParameters<S, R, T> fitResult_)
    {
        final EntropyAnalysis ea = _calc.computeEntropy(fitResult_);
        final double entropy = ea.getEntropy();
        LOG.info("Force pushing params onto chain[" + entropy + "]");
        final ParamProgressFrame<S, R, T> frame = new ParamProgressFrame<>(frameName_, fitResult_, entropy, getLatestFrame(), _rowCount);
        _frameList.add(frame);
    }

    public boolean pushVacuousResults(final String frameName_, final ItemParameters<S, R, T> fitResult_)
    {
        final double currentBest = getLogLikelihood();
        final ParamProgressFrame<S, R, T> frame = new ParamProgressFrame<>(frameName_, fitResult_, currentBest, getLatestFrame(), _rowCount);
        _frameList.add(frame);
        return true;
    }

    public boolean pushResults(final String frameName_, final ItemParameters<S, R, T> fitResult_)
    {
        final double entropy = _calc.computeEntropy(fitResult_).getEntropy();
        return pushResults(frameName_, fitResult_, entropy);
    }

    public boolean pushResults(final String frameName_, final ItemParameters<S, R, T> fitResult_, final double logLikelihood_)
    {
        final double currentBest = getLogLikelihood();

        this.validate(fitResult_, logLikelihood_);

//        if (this.isValidate())
//        {
//            //Since the claim is that the LL improved, let's see if that's true...
//            final EntropyAnalysis ea = _calc.computeEntropy(fitResult_);
//            final double entropy = ea.getEntropy();
//
//            final int compare = MathFunctions.doubleCompareRounded(entropy, logLikelihood_);
//
//            if (compare != 0)
//            {
//                LOG.info("Found entropy mismatch.");
//            }
//        }
        final int prevParamCount = this.getBestParameters().getEffectiveParamCount();
        final int proposedParamCount = fitResult_.getEffectiveParamCount();

        final double aicDifference = MathFunctions.computeAicDifference(prevParamCount, proposedParamCount, currentBest, logLikelihood_, _rowCount);

        // These are negative log likelihoods (positive numbers), a lower number is better.
        // So compare must be < 0, best must be more than the new value.
        final int compare = MathFunctions.doubleCompareRounded(currentBest, logLikelihood_);

        if (compare >= 0)
        {
            LOG.info("Discarding results, likelihood did not improve[" + aicDifference + "]: " + currentBest + " -> " + logLikelihood_);
            return false;
        }

        LOG.info("Log Likelihood improvement[" + frameName_ + "][" + aicDifference + "]: " + currentBest + " -> " + logLikelihood_);

        if (aicDifference >= -5.0)
        {
            LOG.info("Insufficient AIC, discarding results: " + aicDifference);
            return false;
        }

        //This is an improvement. 
        final ParamProgressFrame<S, R, T> frame = new ParamProgressFrame<>(frameName_, fitResult_, logLikelihood_, getLatestFrame(), _rowCount);
        _frameList.add(frame);

        LOG.info("Current chain: " + this.toString());

        return true;
    }

    public int getRowCount()
    {
        return _rowCount;
    }

    public boolean isValidate()
    {
        return _validate;
    }

    public EntropyCalculator<S, R, T> getCalculator()
    {
        return _calc;
    }

    public int size()
    {
        return _frameList.size();
    }

    public double getLogLikelihood()
    {
        return getLatestFrame().getCurrentLogLikelihood();
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();

        builder.append(this.getClass().getName());
        builder.append(" {\n chainName: " + _chainName + " \n");
        builder.append(" size: " + this._frameList.size());

        for (int i = 0; i < _frameList.size(); i++)
        {
            builder.append("\n\t frame[" + i + "]: " + _frameList.get(i).toString());
        }

        builder.append("\n}");

        return builder.toString();
    }

    public ItemParameters<S, R, T> getBestParameters()
    {
        return getLatestFrame().getCurrentParams();
    }

    /**
     * Gets the fitting results taking the start of the chain as the starting
     * point. So how much total progress was made over the course of the whole
     * chain.
     *
     * @return
     */
    public ParamFitResult<S, R, T> getConsolidatedResults()
    {
        final ParamProgressFrame<S, R, T> startFrame = _frameList.get(0);
        final ParamProgressFrame<S, R, T> endFrame = getLatestFrame();

        final ParamFitResult<S, R, T> output = new ParamFitResult<>(startFrame.getCurrentParams(), endFrame.getCurrentParams(), endFrame.getCurrentLogLikelihood(), startFrame.getCurrentLogLikelihood(), _rowCount);
        return output;
    }

    public ParamFitResult<S, R, T> getLatestResults()
    {
        return getLatestFrame().getFitResults();
    }

    public ParamProgressFrame<S, R, T> getLatestFrame()
    {
        return _frameListReadOnly.get(_frameListReadOnly.size() - 1);
    }

    public List<ParamProgressFrame<S, R, T>> getFrameList()
    {
        return _frameListReadOnly;
    }

    public final class ParamProgressFrame<S1 extends ItemStatus<S1>, R1 extends ItemRegressor<R1>, T1 extends ItemCurveType<T1>>
    {
        private final ItemParameters<S1, R1, T1> _current;
        private final double _currentLL;
        private final ParamProgressFrame<S1, R1, T1> _startingPoint;
        private final ParamFitResult<S1, R1, T1> _fitResult;
        private final long _entryTime;
        private final String _frameName;

        private ParamProgressFrame(final String frameName_, final ItemParameters<S1, R1, T1> current_, final double currentLL_, final ParamProgressFrame<S1, R1, T1> startingPoint_, final int rowCount_)
        {
            if (null == current_)
            {
                throw new NullPointerException("Parameters cannot be null.");
            }
            if (Double.isNaN(currentLL_) || Double.isInfinite(currentLL_) || currentLL_ < 0.0)
            {
                throw new IllegalArgumentException("Log Likelihood must be well defined.");
            }

            _frameName = frameName_;
            _current = current_;
            _currentLL = currentLL_;

            if (null == startingPoint_)
            {
                //This is basically a loopback fit result, but it's properly formed, so should be OK.
                _fitResult = new ParamFitResult<>(current_, current_, currentLL_, currentLL_, _rowCount);
                _startingPoint = this;
            }
            else
            {
                _fitResult = new ParamFitResult<>(startingPoint_.getCurrentParams(), current_, currentLL_, startingPoint_.getCurrentLogLikelihood(), _rowCount);
                _startingPoint = startingPoint_;
            }

            _entryTime = System.currentTimeMillis();
        }

        public long getElapsed()
        {
            final long prevEntry = _startingPoint.getEntryTime();
            final long elapsed = _entryTime - prevEntry;
            return elapsed;
        }

        public long getEntryTime()
        {
            return _entryTime;
        }

        public ParamFitResult<S1, R1, T1> getFitResults()
        {
            return _fitResult;
        }

        public ItemParameters<S1, R1, T1> getCurrentParams()
        {
            return _current;
        }

        public double getCurrentLogLikelihood()
        {
            return _currentLL;
        }

        public ParamProgressFrame<S1, R1, T1> getStartingPoint()
        {
            return _startingPoint;
        }

        public double getAicDiff()
        {
            final int paramCountA = this.getStartingPoint().getCurrentParams().getEffectiveParamCount();
            final int paramCountB = this.getCurrentParams().getEffectiveParamCount();

            final double entropyA = this.getStartingPoint().getCurrentLogLikelihood();
            final double entropyB = this.getCurrentLogLikelihood();

            final double aic = MathFunctions.computeAicDifference(paramCountA, paramCountB, entropyA, entropyB, _rowCount);
            return aic;
        }

        @Override
        public String toString()
        {
            final StringBuilder builder = new StringBuilder();

            builder.append("frame[" + _frameName + "][" + this.getAicDiff() + "] {" + _currentLL + ", " + this.getElapsed() + "}");

            return builder.toString();
        }

    }

}
