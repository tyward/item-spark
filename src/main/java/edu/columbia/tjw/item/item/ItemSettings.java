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
package edu.columbia.tjw.item;

import edu.columbia.tjw.item.util.random.RandomTool;
import java.io.Serializable;
import java.util.Random;

/**
 *
 * Some general settings that control how curves are constructed and fit.
 *
 * Generally, don't change these unless you know what you're doing.
 *
 * Also, all the members are final so that this thing is threadsafe. Use the
 * builder to make adjusted versions of this class.
 *
 * One word of warning, these will share a single Random, typically this will
 * still be threadsafe, but not always. Be careful about that.
 *
 * @author tyler
 */
public final class ItemSettings implements Serializable
{
    private static final long serialVersionUID = 0x2bc96e9485221bf6L;

    private static final double STANDARD_AIC_CUTOFF = -5.0;
    private static final int BLOCK_SIZE = 200 * 1000;
    private static final int THREAD_BLOCK_SIZE = 10 * 1000;
    private static final int POLISH_MULTI_START_POINTS = 20;
    private static final boolean ALLOW_INTERACTION_CURVES = true;
    private static final boolean DEFAULT_USE_THREADING = true;
    private static final boolean DEFAULT_POLISH_STARTING_PARAMS = true;
    private static final boolean CENTRALITY_BOUND = true;
    private static final double Z_SCORE_CUTOFF = 1.0;
    private static final boolean DEFAULT_VALIDATE = true;

    private static final ItemSettings DEFAULT = new ItemSettings();

    private final Random _rand;
    private final boolean _randomShuffle;
    private final boolean _useThreading;
    private final boolean _approximateDerivatives;
    private final boolean _polishStartingParams;
    private final boolean _allowInteractionCurves;
    private final boolean _boundCentrality;
    private final int _polishMultStartPoints;

    private final int _minCalibrationCount;
    private final double _improvementRatio;
    private final double _exhaustiveImprovementLimit;

    private final double _aicCutoff;

    //The minimum Z-score such that we will consider two results different. 
    // i.e. two points must differ by at least _zScoreCutoff std deviations to 
    // be considered meaningfully different. Use a number between 1 - 5 here. 1.0 
    // for minimal certainty, 5.0 for proof strong enough to be considered a scientific discovery... (less than 1 in a million to happen by chance)
    private final double _zScoreCutoff;

    private final int _blockSize;
    private final int _threadBlockSize;

    private final boolean _validate;

    public ItemSettings()
    {
        _rand = RandomTool.getRandom();
        _randomShuffle = true;
        _useThreading = DEFAULT_USE_THREADING;
        _approximateDerivatives = false;
        _polishStartingParams = DEFAULT_POLISH_STARTING_PARAMS;
        _allowInteractionCurves = ALLOW_INTERACTION_CURVES;
        _boundCentrality = CENTRALITY_BOUND;
        _polishMultStartPoints = POLISH_MULTI_START_POINTS;
        _minCalibrationCount = 1;
        _improvementRatio = 0.2;
        _exhaustiveImprovementLimit = 0.05;
        _aicCutoff = STANDARD_AIC_CUTOFF;
        _zScoreCutoff = Z_SCORE_CUTOFF;
        _blockSize = BLOCK_SIZE;
        _threadBlockSize = THREAD_BLOCK_SIZE;
        _validate = DEFAULT_VALIDATE;
    }

    public ItemSettings(final ItemSettingsBuilder builder_)
    {
        _rand = builder_.getRand();
        _randomShuffle = builder_.isRandomShuffle();
        _useThreading = builder_.isUseThreading();
        _approximateDerivatives = builder_.isApproximateDerivatives();
        _polishStartingParams = builder_.isPolishStartingParams();
        _allowInteractionCurves = builder_.isAllowInteractionCurves();
        _boundCentrality = builder_.isBoundCentrality();
        _polishMultStartPoints = builder_.getPolishMultStartPoints();
        _minCalibrationCount = builder_.getMinCalibrationCount();
        _improvementRatio = builder_.getImprovementRatio();
        _exhaustiveImprovementLimit = builder_.getExhaustiveImprovementLimit();
        _aicCutoff = builder_.getAicCutoff();
        _zScoreCutoff = builder_.getzScoreCutoff();
        _blockSize = builder_.getBlockSize();
        _threadBlockSize = builder_.getThreadBlockSize();
        _validate = builder_.isValidate();
    }

    public double getExhaustiveImprovementLimit()
    {
        return _exhaustiveImprovementLimit;
    }

    public int getCalibrateSize()
    {
        return _minCalibrationCount;
    }

    public double getImprovementRatio()
    {
        return _improvementRatio;
    }

    public boolean getBoundCentrality()
    {
        return _boundCentrality;
    }

    public int getPolishMultiStartPoints()
    {
        return _polishMultStartPoints;
    }

    public boolean getDoValidate()
    {
        return _validate;
    }

    public boolean getAllowInteractionCurves()
    {
        return _allowInteractionCurves;
    }

    public boolean getPolishStartingParams()
    {
        return _polishStartingParams;
    }

    public boolean getUseThreading()
    {
        return _useThreading;
    }

    public int getBlockSize()
    {
        return _blockSize;
    }

    public int getThreadBlockSize()
    {
        return _threadBlockSize;
    }

    public double getAicCutoff()
    {
        return _aicCutoff;
    }

    public boolean isRandomShuffle()
    {
        return _randomShuffle;
    }

    public Random getRandom()
    {
        return _rand;
    }

    public double getZScoreCutoff()
    {
        return _zScoreCutoff;
    }

    public boolean approximateDerivatives()
    {
        return _approximateDerivatives;
    }

    public ItemSettingsBuilder makeBuilder()
    {
        return new ItemSettingsBuilder(this);
    }

    public final class ItemSettingsBuilder
    {
        private Random _rand;
        private boolean _randomShuffle;
        private boolean _useThreading;
        private boolean _approximateDerivatives;
        private boolean _polishStartingParams;
        private boolean _allowInteractionCurves;
        private boolean _boundCentrality;
        private int _polishMultStartPoints;
        private int _minCalibrationCount;
        private double _improvementRatio;
        private double _exhaustiveImprovementLimit;
        private double _aicCutoff;
        private double _zScoreCutoff;
        private int _blockSize;
        private int _threadBlockSize;
        private boolean _validate;

        public ItemSettingsBuilder()
        {
            this(DEFAULT);
        }

        public ItemSettingsBuilder(final ItemSettings base_)
        {
            _rand = base_.getRandom();
            _randomShuffle = base_.isRandomShuffle();
            _useThreading = base_.getUseThreading();
            _approximateDerivatives = base_.approximateDerivatives();
            _polishStartingParams = base_.getPolishStartingParams();
            _allowInteractionCurves = base_.getAllowInteractionCurves();
            _boundCentrality = base_.getBoundCentrality();
            _polishMultStartPoints = base_.getPolishMultiStartPoints();
            _minCalibrationCount = base_.getCalibrateSize();
            _improvementRatio = base_.getImprovementRatio();
            _exhaustiveImprovementLimit = base_.getExhaustiveImprovementLimit();
            _aicCutoff = base_.getAicCutoff();
            _zScoreCutoff = base_.getZScoreCutoff();
            _blockSize = base_.getBlockSize();
            _threadBlockSize = base_.getThreadBlockSize();
            _validate = base_.getDoValidate();
        }

        public ItemSettings build()
        {
            return new ItemSettings(this);
        }

        public Random getRand()
        {
            return _rand;
        }

        public boolean isRandomShuffle()
        {
            return _randomShuffle;
        }

        public boolean isUseThreading()
        {
            return _useThreading;
        }

        public boolean isApproximateDerivatives()
        {
            return _approximateDerivatives;
        }

        public boolean isPolishStartingParams()
        {
            return _polishStartingParams;
        }

        public boolean isAllowInteractionCurves()
        {
            return _allowInteractionCurves;
        }

        public boolean isBoundCentrality()
        {
            return _boundCentrality;
        }

        public int getPolishMultStartPoints()
        {
            return _polishMultStartPoints;
        }

        public int getMinCalibrationCount()
        {
            return _minCalibrationCount;
        }

        public double getImprovementRatio()
        {
            return _improvementRatio;
        }

        public double getExhaustiveImprovementLimit()
        {
            return _exhaustiveImprovementLimit;
        }

        public double getAicCutoff()
        {
            return _aicCutoff;
        }

        public double getzScoreCutoff()
        {
            return _zScoreCutoff;
        }

        public int getBlockSize()
        {
            return _blockSize;
        }

        public int getThreadBlockSize()
        {
            return _threadBlockSize;
        }

        public boolean isValidate()
        {
            return _validate;
        }

        public ItemSettingsBuilder setRand(Random _rand)
        {
            this._rand = _rand;
            return this;
        }

        public ItemSettingsBuilder setRandomShuffle(boolean _randomShuffle)
        {
            this._randomShuffle = _randomShuffle;
            return this;
        }

        public ItemSettingsBuilder setUseThreading(boolean _useThreading)
        {
            this._useThreading = _useThreading;
            return this;
        }

        public ItemSettingsBuilder setApproximateDerivatives(boolean _approximateDerivatives)
        {
            this._approximateDerivatives = _approximateDerivatives;
            return this;
        }

        public ItemSettingsBuilder setPolishStartingParams(boolean _polishStartingParams)
        {
            this._polishStartingParams = _polishStartingParams;
            return this;
        }

        public ItemSettingsBuilder setAllowInteractionCurves(boolean _allowInteractionCurves)
        {
            this._allowInteractionCurves = _allowInteractionCurves;
            return this;
        }

        public ItemSettingsBuilder setBoundCentrality(boolean _boundCentrality)
        {
            this._boundCentrality = _boundCentrality;
            return this;
        }

        public ItemSettingsBuilder setPolishMultStartPoints(int _polishMultStartPoints)
        {
            this._polishMultStartPoints = _polishMultStartPoints;
            return this;
        }

        public ItemSettingsBuilder setMinCalibrationCount(int _minCalibrationCount)
        {
            this._minCalibrationCount = _minCalibrationCount;
            return this;
        }

        public ItemSettingsBuilder setImprovementRatio(double _improvementRatio)
        {
            this._improvementRatio = _improvementRatio;
            return this;
        }

        public ItemSettingsBuilder setExhaustiveImprovementLimit(double _exhaustiveImprovementLimit)
        {
            this._exhaustiveImprovementLimit = _exhaustiveImprovementLimit;
            return this;
        }

        public ItemSettingsBuilder setAicCutoff(double aicCutoff_)
        {
            if (aicCutoff_ > 0.0)
            {
                throw new IllegalArgumentException("The AIC cutoff must be negative: " + aicCutoff_);
            }

            this._aicCutoff = aicCutoff_;
            return this;
        }

        public ItemSettingsBuilder setzScoreCutoff(double zScoreCutoff_)
        {
            this._zScoreCutoff = zScoreCutoff_;
            return this;
        }

        public ItemSettingsBuilder setBlockSize(int blockSize_)
        {
            this._blockSize = blockSize_;
            return this;
        }

        public ItemSettingsBuilder setThreadBlockSize(int threadBlockSize_)
        {
            if (threadBlockSize_ < 100)
            {
                throw new IllegalArgumentException("Thread block size should not be too small: " + threadBlockSize_);
            }

            if (this._blockSize < threadBlockSize_)
            {
                _blockSize = threadBlockSize_;
            }

            this._threadBlockSize = threadBlockSize_;
            return this;
        }

        public ItemSettingsBuilder setValidate(boolean _validate)
        {
            this._validate = _validate;
            return this;
        }

    }

}
