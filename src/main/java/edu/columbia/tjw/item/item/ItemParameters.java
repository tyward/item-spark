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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 *
 * @author tyler
 * @param <S> The status type for this model
 * @param <R> The regressor type for this model
 * @param <T> The curve type for this model
 */
public final class ItemParameters<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements Serializable
{
    //This is just to make it clear that the 0th entry is always the intercept.
    private static final int INTERCEPT_INDEX = 0;
    private static final long serialVersionUID = 0x35c74a5424d6cf48L;

    private final S _status;
    private final int _selfIndex;
    private final R _intercept;
    private final List<ItemCurve<T>> _trans;
    private final List<ParamFilter<S, R, T>> _filters;
    private final UniqueBetaFilter _uniqFilter = new UniqueBetaFilter();

    private final List<R> _uniqueFields;

    private final double[][] _betas;

    //This is 2-D so that we can have multiple field/curves per entry, allowing for interactions through weighting.
    private final int[][] _fieldOffsets;
    private final int[][] _transOffsets;

    // If  we allow only one beta to be set for this entry, place its index here, otherwise -1.
    private final int[] _uniqueBeta;

    private final int _effectiveParamCount;

    public ItemParameters(final S status_, final R intercept_)
    {
        if (null == status_)
        {
            throw new NullPointerException("Status cannot be null.");
        }
        if (null == intercept_)
        {
            throw new NullPointerException("Intercept cannot be null.");
        }

        _status = status_;
        _intercept = intercept_;

        _betas = new double[status_.getReachableCount()][1];

        _uniqueFields = Collections.unmodifiableList(Collections.singletonList(intercept_));
        _trans = Collections.unmodifiableList(Collections.singletonList(null));
        _filters = Collections.unmodifiableList(Collections.emptyList());

        _uniqueBeta = new int[1];
        _uniqueBeta[INTERCEPT_INDEX] = -1;

        _fieldOffsets = new int[1][1];
        _fieldOffsets[INTERCEPT_INDEX][0] = 0;

        _transOffsets = new int[1][1];
        _transOffsets[INTERCEPT_INDEX][0] = 0;

        _selfIndex = _status.getReachable().indexOf(_status);
        _effectiveParamCount = calculateEffectiveParamCount();
    }

    /**
     * The constructor used to change betas, or add filters.
     *
     * @param base_
     * @param betas_
     * @param addedFilters_
     */
    private ItemParameters(final ItemParameters<S, R, T> base_, final double[][] betas_, final Collection<ParamFilter<S, R, T>> addedFilters_)
    {
        _status = base_._status;
        _intercept = base_._intercept;
        _selfIndex = base_._selfIndex;
        _trans = base_._trans;
        _uniqueFields = base_._uniqueFields;
        _fieldOffsets = base_._fieldOffsets;
        _transOffsets = base_._transOffsets;
        _uniqueBeta = base_._uniqueBeta;

        if (null != betas_)
        {
            final int size = base_._betas.length;

            //We will build up a clone to make sure there are no problems with external modification.
            double[][] newBeta = new double[size][];

            if (size != betas_.length)
            {
                throw new IllegalArgumentException("Beta matrix is the wrong size.");
            }
            for (int i = 0; i < size; i++)
            {
                if (base_._betas[i].length != betas_[i].length)
                {
                    throw new IllegalArgumentException("Beta matrix is the wrong size.");
                }

                newBeta[i] = betas_[i].clone();
            }

            _betas = newBeta;
        }
        else
        {
            _betas = base_._betas;
        }

        if (addedFilters_.size() < 1)
        {
            _filters = base_._filters;
        }
        else
        {
            final List<ParamFilter<S, R, T>> newFilters = new ArrayList<>(base_._filters);

            for (final ParamFilter<S, R, T> next : addedFilters_)
            {
                newFilters.add(next);
            }

            _filters = Collections.unmodifiableList(newFilters);
        }

        _effectiveParamCount = calculateEffectiveParamCount();
    }

    /**
     * Used to make a new set of parameters with a new entry.
     *
     * @param base_
     * @param regs_
     * @param curves_
     */
    private ItemParameters(final ItemParameters<S, R, T> base_, final ItemCurveParams<R, T> curveParams_, final S toStatus_)
    {
        _status = base_._status;
        _intercept = base_._intercept;
        _selfIndex = base_._selfIndex;
        _filters = base_._filters;

        final SortedSet<R> newFields = new TreeSet<>();

        //Workaround for issues with nulls.
        final Set<ItemCurve<T>> newTrans = new HashSet<>();

        newFields.addAll(base_._uniqueFields);
        newFields.addAll(curveParams_.getRegressors());

        newTrans.addAll(base_._trans);
        newTrans.addAll(curveParams_.getCurves());

        //Always add the new entry to the end...
        final int baseEntryCount = base_.getEntryCount();
        final int newEntryCount = baseEntryCount + 1;
        final int endIndex = baseEntryCount;

        //Just add the new entry to the end of the list.
        _trans = Collections.unmodifiableList(new ArrayList<>(newTrans));
        _uniqueFields = Collections.unmodifiableList(new ArrayList<>(newFields));

        _uniqueBeta = Arrays.copyOf(base_._uniqueBeta, newEntryCount);

        final int toIndex;

        if (null == toStatus_)
        {
            toIndex = -1;
        }
        else
        {
            toIndex = _status.getReachable().indexOf(toStatus_);

            if (toIndex == -1)
            {
                throw new IllegalArgumentException("Not reachable status: " + toStatus_);
            }
        }

        _uniqueBeta[endIndex] = toIndex;
        _fieldOffsets = new int[newEntryCount][];
        _transOffsets = new int[newEntryCount][];
        _betas = new double[base_._betas.length][];

        //Copy over prev betas, fill out with zeros. Will adjust again to change the betas...
        for (int i = 0; i < _betas.length; i++)
        {
            _betas[i] = Arrays.copyOf(base_._betas[i], newEntryCount);
        }

        //First, pull in all the old entries.
        for (int i = 0; i < baseEntryCount; i++)
        {
            final int depth = base_.getEntryDepth(i);
            _fieldOffsets[i] = new int[depth];
            _transOffsets[i] = new int[depth];

            for (int w = 0; w < depth; w++)
            {
                final R field = base_.getEntryRegressor(i, w);
                final ItemCurve<T> curve = base_.getEntryCurve(i, w);

                _fieldOffsets[i][w] = _uniqueFields.indexOf(field);
                _transOffsets[i][w] = _trans.indexOf(curve);
            }
        }

        //Now fill out the last entry...
        final int endDepth = curveParams_.getEntryDepth();

        _fieldOffsets[endIndex] = new int[endDepth];
        _transOffsets[endIndex] = new int[endDepth];

        for (int i = 0; i < endDepth; i++)
        {
            final R field = curveParams_.getRegressor(i);
            final ItemCurve<T> curve = curveParams_.getCurve(i);

            _fieldOffsets[endIndex][i] = _uniqueFields.indexOf(field);
            _transOffsets[endIndex][i] = _trans.indexOf(curve);
        }

        if (toIndex != -1)
        {
            //In this case, we have a real curve, let's set the betas....
            _betas[toIndex][endIndex] = curveParams_.getBeta();
            _betas[toIndex][INTERCEPT_INDEX] += curveParams_.getIntercept();
        }

        _effectiveParamCount = calculateEffectiveParamCount();
    }

    /**
     * Used to make a new set of parameters with some indices dropped.
     *
     * @param base_
     * @param dropIndices_
     */
    private ItemParameters(final ItemParameters<S, R, T> base_, final int[] dropIndices_)
    {
        _status = base_._status;
        _intercept = base_._intercept;
        _selfIndex = base_._selfIndex;
        _filters = base_._filters;

        final int startSize = base_.getEntryCount();
        final boolean[] drop = new boolean[startSize];

        // We are open to the possibility that dropIndices_ is not well formed, 
        // and some indices occur multiple times.
        for (int i = 0; i < dropIndices_.length; i++)
        {
            drop[dropIndices_[i]] = true;
        }

        if (drop[INTERCEPT_INDEX])
        {
            throw new IllegalArgumentException("The intercept index cannot be dropped.");
        }

        int dropped = 0;

        for (int i = 0; i < startSize; i++)
        {
            if (drop[i])
            {
                dropped++;
            }
        }

        final int newSize = startSize - dropped;

        _fieldOffsets = new int[newSize][];
        _transOffsets = new int[newSize][];
        _betas = new double[base_._betas.length][newSize];
        _uniqueBeta = new int[newSize];

        final SortedSet<R> newFields = new TreeSet<>();
        final Set<ItemCurve<T>> newTrans = new HashSet<>();

        int pointer = 0;

        for (int i = 0; i < startSize; i++)
        {
            if (drop[i])
            {
                continue;
            }

            _uniqueBeta[pointer] = base_._uniqueBeta[i];

            final int depth = base_.getEntryDepth(i);

            _fieldOffsets[pointer] = new int[depth];
            _transOffsets[pointer] = new int[depth];
            pointer++;

            for (int w = 0; w < depth; w++)
            {
                final R next = base_.getEntryRegressor(i, w);
                final ItemCurve<T> nextCurve = base_.getEntryCurve(i, w);

                newFields.add(next);
                newTrans.add(nextCurve);
            }
        }

        if (pointer != newSize)
        {
            throw new IllegalStateException("Impossible.");
        }

        this._trans = Collections.unmodifiableList(new ArrayList<>(newTrans));
        this._uniqueFields = Collections.unmodifiableList(new ArrayList<>(newFields));

        pointer = 0;

        for (int i = 0; i < startSize; i++)
        {
            if (drop[i])
            {
                continue;
            }

            final int depth = base_.getEntryDepth(i);

            for (int w = 0; w < depth; w++)
            {
                final R reg = base_.getEntryRegressor(i, w);
                final ItemCurve<T> curve = base_.getEntryCurve(i, w);

                final int rIndex = _uniqueFields.indexOf(reg);
                final int cIndex = _trans.indexOf(curve);

                if (rIndex < 0 || cIndex < 0)
                {
                    throw new IllegalStateException("Impossible.");
                }

                _fieldOffsets[pointer][w] = rIndex;
                _transOffsets[pointer][w] = cIndex;
            }

            for (int w = 0; w < _betas.length; w++)
            {
                _betas[w][pointer] = base_._betas[w][i];
            }

            pointer++;
        }

        _effectiveParamCount = calculateEffectiveParamCount();
    }

    private int calculateEffectiveParamCount()
    {
        final int effectiveTransSize = this._status.getReachableCount() - 1;
        int paramCount = 0;

        for (int i = 0; i < this.getEntryCount(); i++)
        {
            final boolean isRestricted = (this.getEntryStatusRestrict(i) != null);

            if (isRestricted)
            {
                paramCount += 1;
            }
            else
            {
                paramCount += effectiveTransSize;
            }

            for (int z = 0; z < this.getEntryDepth(i); z++)
            {
                final ItemCurve<T> curve = this.getEntryCurve(i, z);

                if (null == curve)
                {
                    continue;
                }

                paramCount += curve.getCurveType().getParamCount();
            }
        }

        return paramCount;
    }

    public int getEffectiveParamCount()
    {
        return _effectiveParamCount;
    }

    public int getInterceptIndex()
    {
        return INTERCEPT_INDEX;
    }

    public int getEntryCount()
    {
        return _fieldOffsets.length;
    }

    public int getEntryDepth(final int entryIndex_)
    {
        return _fieldOffsets[entryIndex_].length;
    }

    public S getEntryStatusRestrict(final int entryIndex_)
    {
        final int uniqueBeta = _uniqueBeta[entryIndex_];

        if (uniqueBeta == -1)
        {
            return null;
        }

        return this._status.getReachable().get(uniqueBeta);
    }

    public int getEntryRegressorOffset(final int entryIndex_, final int entryDepth_)
    {
        return _fieldOffsets[entryIndex_][entryDepth_];
    }

    public R getEntryRegressor(final int entryIndex_, final int entryDepth_)
    {
        final int offset = getEntryRegressorOffset(entryIndex_, entryDepth_);
        return _uniqueFields.get(offset);
    }

    public int getEntryCurveOffset(final int entryIndex_, final int entryDepth_)
    {
        return _transOffsets[entryIndex_][entryDepth_];
    }

    /**
     * This function will find the entry corresponding to the given curve
     * parameters.
     *
     * HOWEVER, it is necessary for the entry to match exactly, meaning many of
     * the constituent objects must be the exact same object. This is most
     * useful for (for instance) finding the entry of some ItemCurveParams that
     * were just used a moment ago to expand these parameters.
     *
     * @param params_ The ItemCurveParams to look for.
     * @return The entryIndex that would return equivalent params in response to
     * getEntryCurveParams(index), -1 if no such entry.
     */
    public int getEntryIndex(final ItemCurveParams<R, T> params_)
    {
        final int entryCount = this.getEntryCount();
        final int testDepth = params_.getEntryDepth();

        outer:
        for (int i = 0; i < entryCount; i++)
        {
            final int depth = this.getEntryDepth(i);

            if (depth != testDepth)
            {
                continue;
            }

            for (int w = 0; w < depth; w++)
            {
                if (this.getEntryCurve(i, w) != params_.getCurve(w))
                {
                    continue outer;
                }
                if (this.getEntryRegressor(i, w) != params_.getRegressor(w))
                {
                    continue outer;
                }

                //Do I verify the value of beta as well? No, I think I can safely skip that.
            }

            //We made it, this is the entry you're looking for.
            return i;
        }

        return -1;
    }

    public ItemCurveParams<R, T> getEntryCurveParams(final int entryIndex_)
    {
        return getEntryCurveParams(entryIndex_, false);
    }

    public ItemCurveParams<R, T> getEntryCurveParams(final int entryIndex_, final boolean allowNonCurve_)
    {
        final int toIndex = _uniqueBeta[entryIndex_];

        if (toIndex == -1 && !allowNonCurve_)
        {
            throw new IllegalArgumentException("Can only extract curve params for actual curves.");
        }

        final int depth = getEntryDepth(entryIndex_);
        final List<R> regs = new ArrayList<>(depth);
        final List<ItemCurve<T>> curves = new ArrayList<>(depth);

        for (int i = 0; i < depth; i++)
        {
            final R reg = this.getEntryRegressor(entryIndex_, i);
            final ItemCurve<T> curve = this.getEntryCurve(entryIndex_, i);
            regs.add(reg);
            curves.add(curve);
        }

        final double interceptAdjustment = 0.0;
        final double beta;

        if (toIndex != -1)
        {
            beta = _betas[toIndex][entryIndex_];
        }
        else
        {
            beta = 0.0;
        }

        final ItemCurveParams<R, T> output = new ItemCurveParams<>(interceptAdjustment, beta, regs, curves);
        return output;
    }

    public ItemCurve<T> getEntryCurve(final int entryIndex_, final int entryDepth_)
    {
        final int offset = getEntryCurveOffset(entryIndex_, entryDepth_);
        return _trans.get(offset);
    }

    public List<R> getUniqueRegressors()
    {
        return _uniqueFields;
    }

    public List<ItemCurve<T>> getUniqueCurves()
    {
        return _trans;
    }

    public boolean betaIsFrozen(S toStatus_, int paramEntry_, final Collection<ParamFilter<S, R, T>> otherFilters_)
    {
        if (_uniqFilter.betaIsFrozen(this, toStatus_, paramEntry_))
        {
            return true;
        }

        for (final ParamFilter<S, R, T> next : getFilters())
        {
            if (next.betaIsFrozen(this, toStatus_, paramEntry_))
            {
                return true;
            }
        }

        if (null == otherFilters_)
        {
            return false;
        }

        for (final ParamFilter<S, R, T> next : otherFilters_)
        {
            if (next.betaIsFrozen(this, toStatus_, paramEntry_))
            {
                return true;
            }
        }

        return false;
    }

    public boolean curveIsForbidden(S toStatus_, ItemCurveParams<R, T> curveParams_, final Collection<ParamFilter<S, R, T>> otherFilters_)
    {
        if (_uniqFilter.curveIsForbidden(this, toStatus_, curveParams_))
        {
            return true;
        }

        for (final ParamFilter<S, R, T> next : getFilters())
        {
            if (next.curveIsForbidden(this, toStatus_, curveParams_))
            {
                return true;
            }
        }

        if (null == otherFilters_)
        {
            return false;
        }

        for (final ParamFilter<S, R, T> next : otherFilters_)
        {
            if (next.curveIsForbidden(this, toStatus_, curveParams_))
            {
                return true;
            }
        }

        return false;
    }

    private List<ParamFilter<S, R, T>> getFilters()
    {
        return _filters;
    }

    private boolean compareTrans(final ItemCurve<?> trans1_, final ItemCurve<?> trans2_)
    {
        //We demand exact reference equality for this operation.
        //return (trans1_ == trans2_);
        if (trans1_ == trans2_)
        {
            return true;
        }
        if (null == trans1_)
        {
            return false;
        }

        return trans1_.equals(trans2_);
    }

    public ItemParameters<S, R, T> dropRegressor(final R field_)
    {
        final int regCount = this.getEntryCount();
        final boolean[] keep = new boolean[regCount];

        outer:
        for (int i = 0; i < regCount; i++)
        {
            final R next = getEntryRegressor(i, 0);

            if (!field_.equals(next))
            {
                keep[i] = true;
                continue;
            }

            keep[i] = false;
        }

        return dropEntries(keep);
    }

    public ItemParameters<S, R, T> dropIndex(final int index_)
    {
        final int regCount = this.getEntryCount();
        final boolean[] keep = new boolean[regCount];
        Arrays.fill(keep, true);
        keep[index_] = false;

        return dropEntries(keep);
    }

    private ItemParameters<S, R, T> dropEntries(final boolean[] keep_)
    {
        int dropCount = 0;

        for (final boolean next : keep_)
        {
            if (!next)
            {
                dropCount++;
            }
        }

        final int[] dropIndices = new int[dropCount];
        int pointer = 0;

        for (int i = 0; i < keep_.length; i++)
        {
            if (keep_[i])
            {
                continue;
            }

            dropIndices[pointer++] = i;
        }

        return new ItemParameters<>(this, dropIndices);
    }

    /**
     * Adds an empty beta entry, with just the given field.
     *
     * Note, if these parameters already contain a raw flag for this beta, then
     * this is returned unchanged.
     *
     * @param regressor_
     * @return
     */
    public ItemParameters<S, R, T> addBeta(final R regressor_)
    {
        for (int i = 0; i < this.getEntryCount(); i++)
        {
            if (this.getEntryDepth(i) != 1)
            {
                continue;
            }

            if (this.getEntryRegressor(i, 0) == regressor_)
            {
                //Already have a flag for this regressor, return unchanged.
                return this;
            }
        }

        final ItemCurveParams<R, T> fieldParams = new ItemCurveParams<>(0.0, 0.0, regressor_, null);
        return new ItemParameters<>(this, fieldParams, null);
    }

    /**
     * Creates a new set of parameters with an additional beta.
     *
     * @param curveParams_
     * @param toStatus_
     *
     * @return A new set of parameters, with the given curve added and its beta
     * set to zero.
     */
    public ItemParameters<S, R, T> addBeta(final ItemCurveParams<R, T> curveParams_, final S toStatus_)
    {
        if (null == toStatus_)
        {
            for (final ItemCurve<T> curve : curveParams_.getCurves())
            {
                if (null != curve)
                {
                    //This is OK as long as none of the curves are set (in which case, it's just a list of flags...)
                    throw new NullPointerException("To status cannot be null for non-flag curves.");
                }
            }
        }
        if (toStatus_ == this._status)
        {
            throw new NullPointerException("Beta for from status must be zero.");
        }

        return new ItemParameters<>(this, curveParams_, toStatus_);
    }

    public ItemParameters<S, R, T> updateBetas(final double[][] betas_)
    {
        return new ItemParameters<>(this, betas_, Collections.emptyList());
    }

    public ItemParameters<S, R, T> addFilters(final Collection<ParamFilter<S, R, T>> filters_)
    {
        return new ItemParameters<>(this, null, filters_);
    }

    public ItemParameters<S, R, T> addFilter(final ParamFilter<S, R, T> filter_)
    {
        final Set<ParamFilter<S, R, T>> set = Collections.singleton(filter_);
        return this.addFilters(set);
    }

    public int getReachableSize()
    {
        return _status.getReachableCount();
    }

    public int toStatusIndex(final S toStatus_)
    {
        final int index = _status.getReachable().indexOf(toStatus_);
        return index;
    }

    public double getBeta(final int statusIndex_, final int regIndex_)
    {
        return _betas[statusIndex_][regIndex_];
    }

    public double[][] getBetas()
    {
        final double[][] output = _betas.clone();

        for (int i = 0; i < output.length; i++)
        {
            output[i] = output[i].clone();
        }

        return output;
    }

    public S getStatus()
    {
        return _status;
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();
        builder.append("ItemParameters[" + Integer.toHexString(System.identityHashCode(this)) + "][\n");

        builder.append("\tEntries[" + this.getEntryCount() + "]: \n");

        for (int i = 0; i < this.getEntryCount(); i++)
        {
            builder.append("\t\t Entry \n \t\t\tBeta: [");

            for (int w = 0; w < _betas.length; w++)
            {
                if (w > 0)
                {
                    builder.append(", ");
                }

                builder.append(_betas[w][i]);
            }

            builder.append("]\n");

            final int depth = this.getEntryDepth(i);

            builder.append("\t\t\t Entry Definition[" + i + ", depth=" + depth + "]: \n");

            for (int w = 0; w < this.getEntryDepth(i); w++)
            {
                final R reg = this.getEntryRegressor(i, w);
                final ItemCurve<T> curve = this.getEntryCurve(i, w);

                builder.append("\t\t\t[" + i + ", " + w + "]:" + reg + ":" + curve + "\n");
            }

            builder.append("\t\t\t Entry Beta Restricted: " + _uniqueBeta[i] + "\n");
        }

        builder.append("]\n\n");

        return builder.toString();
    }

    private double[] resizeArray(final double[] input_, final int insertionPoint_)
    {
        final double[] output = new double[input_.length + 1];

        for (int i = 0; i < insertionPoint_; i++)
        {
            output[i] = input_[i];
        }

        output[insertionPoint_] = 0.0;

        for (int i = insertionPoint_; i < input_.length; i++)
        {
            output[i + 1] = input_[i];
        }

        return output;
    }

    private int[] resizeArray(final int[] input_, final int insertionPoint_, final int insertionValue_)
    {
        final int[] output = new int[input_.length + 1];

        for (int i = 0; i < insertionPoint_; i++)
        {
            output[i] = input_[i];
        }

        output[insertionPoint_] = insertionValue_;

        for (int i = insertionPoint_; i < input_.length; i++)
        {
            output[i + 1] = input_[i];
        }

        return output;
    }

    private final class UniqueBetaFilter implements ParamFilter<S, R, T>
    {
        private static final long serialVersionUID = 0x49e89a36e4553a69L;

        @Override
        public boolean betaIsFrozen(ItemParameters<S, R, T> params_, S toStatus_, int paramEntry_)
        {
            if (params_.getStatus() == toStatus_)
            {
                return true;
            }

            final S restrict = params_.getEntryStatusRestrict(paramEntry_);

            if (null == restrict)
            {
                return false;
            }

            if (restrict != toStatus_)
            {
                return true;
            }

            return false;
        }

        @Override
        public boolean curveIsForbidden(ItemParameters<S, R, T> params_, S toStatus_, ItemCurveParams<R, T> curveParams_)
        {
            //We don't forbid any new additions, except where fromStatus_ == toStatus_
            return (params_.getStatus() == toStatus_);
        }

    }

}
