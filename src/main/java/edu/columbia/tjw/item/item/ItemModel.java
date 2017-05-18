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

import edu.columbia.tjw.item.fit.ItemParamGrid;
import edu.columbia.tjw.item.fit.ParamFittingGrid;
import edu.columbia.tjw.item.fit.param.ParamFitter;
import edu.columbia.tjw.item.util.LogLikelihood;
import edu.columbia.tjw.item.util.LogUtil;
import edu.columbia.tjw.item.util.MathFunctions;
import edu.columbia.tjw.item.util.MultiLogistic;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 *
 * A class representing an ITEM model.
 *
 * Note that this class is not threadsafe. You need to synchronize on your own,
 * or clone this model and use the clone in other threads.
 *
 * @author tyler
 * @param <S> The status type for this model
 * @param <R> The regressor type for this model
 * @param <T> The curve type for this model
 */
public final class ItemModel<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements Cloneable
{
    private static final Logger LOG = LogUtil.getLogger(ItemModel.class);
    private final double ROUNDING_TOLERANCE = 1.0e-8;
    private final LogLikelihood<S> _likelihood;
    private final ItemParameters<S, R, T> _params;

    private final double[][] _betas;
    private final int _reachableSize;

    private final double[] _rawRegWorkspace;
    private final double[] _regWorkspace;
    private final double[] _probWorkspace;
    private final double[] _actualProbWorkspace;

    /**
     * Create a new item model from its parameters.
     *
     * @param params_ The parameters that define this model
     */
    public ItemModel(final ItemParameters<S, R, T> params_)
    {
        synchronized (this)
        {
            //This is synchronized so that clone may safely be called by any thread. 
            //In this way, we can generate models that are detached form one another, 
            //and may safely be used simultaneously in different threads without requiring
            //any additiona synchronization. 
            _params = params_;
            final S status = params_.getStatus();

            //_reachable = status.getReachable();
            _betas = params_.getBetas();
            _reachableSize = status.getReachableCount();

            _likelihood = new LogLikelihood<>(status);

            final int entryCount = params_.getEntryCount();

            _rawRegWorkspace = new double[params_.getUniqueRegressors().size()];
            _regWorkspace = new double[entryCount];
            _probWorkspace = new double[_reachableSize];
            _actualProbWorkspace = new double[_reachableSize];
        }
    }

    public S getStatus()
    {
        return _params.getStatus();
    }

    public final ItemParameters<S, R, T> getParams()
    {
        return _params;
    }

    /**
     * N.B: This is NOT threadsafe! Generate a new model for each thread, there
     * is some internal workspace associated that can't be shared.
     *
     * @param grid_ The grid holding the data
     * @param index_ The row of the grid on which to apply the model
     * @return The log likelihood of the data point represented by this row
     */
    public final double logLikelihood(final ParamFittingGrid<S, R, T> grid_, final int index_)
    {
        if (!grid_.hasNextStatus(index_))
        {
            return 0.0;
        }

        final double[] computed = _probWorkspace;
        //final double[] actual = workspace_.getActualProbabilityWorkspace();
        transitionProbability(grid_, index_, computed);

        //Arrays.fill(actual, 0.0);
        final int actualOutcome = grid_.getNextStatus(index_);
        final int mapped = _likelihood.ordinalToOffset(actualOutcome);
        //actual[mapped] = 1.0;

        //If this item took a forbidden transition, ignore the data point.
        if (mapped < 0)
        {
            return 0.0;
        }

        final double logLikelihood = _likelihood.logLikelihood(computed, mapped);
        return logLikelihood;
    }

    private void fillWorkspace(final double[] rawRegressors_, final double[] regWorkspace_)
    {
        if (rawRegressors_.length != _rawRegWorkspace.length)
        {
            throw new IllegalArgumentException("Length mismatch.");
        }

        final int numEntries = _params.getEntryCount();

        if (regWorkspace_.length != numEntries)
        {
            throw new IllegalArgumentException("Length mismatch.");
        }

        for (int i = 0; i < numEntries; i++)
        {
            final int depth = _params.getEntryDepth(i);
            double weight = 1.0;

            for (int w = 0; w < depth; w++)
            {
                final int regIndex = _params.getEntryRegressorOffset(i, w);
                final double rawReg = rawRegressors_[regIndex];
                final ItemCurve<T> curve = _params.getEntryCurve(i, w);

                if (null == curve)
                {
                    weight *= rawReg;
                }
                else
                {
                    weight *= curve.transform(rawReg);
                }
            }

            regWorkspace_[i] = weight;
        }
    }

    /**
     * This will take packed probabilities (only the reachable states for this
     * status), and fill out a vector of unpacked probabilities (all statuses
     * are represented).
     *
     * @param packed_ The probabilities for reachable statuses. (input)
     * @param unpacked_ A vector to hold the probabilities for all statuses.
     * Unreachable statuses will get 0.0.
     */
    public void unpackProbabilities(final double[] packed_, final double[] unpacked_)
    {
        final S status = getStatus();
        //final List<S> reachable = status.getReachable();
        final int reachableCount = status.getReachableCount();
        final int statCount = status.getFamily().size();

        if (packed_.length != reachableCount)
        {
            throw new IllegalArgumentException("Input is the wrong size: " + packed_.length);
        }
        if (unpacked_.length != statCount)
        {
            throw new IllegalArgumentException("Output is the wrong size: " + packed_.length);
        }

        Arrays.fill(unpacked_, 0.0);
        double sum = 0.0;

        for (int i = 0; i < statCount; i++)
        {
            final int reachableOrdinal = _likelihood.ordinalToOffset(i);

            if (reachableOrdinal < 0)
            {
                unpacked_[i] = 0.0;
                continue;
            }

            final double prob = packed_[reachableOrdinal];
            sum += prob;
            unpacked_[i] = prob;
        }

        final double diff = Math.abs(sum - 1.0);

        if (!(diff < ROUNDING_TOLERANCE))
        {
            throw new IllegalArgumentException("Rounding tolerance exceeded by probability vector: " + diff);
        }
    }

    /**
     * Compute the derivative of the log likelihood with respect to the given
     * parameters, over the given data set.
     *
     * Note that this will be the SUM of the log likelihoods, meaning that
     * you'll need to normalize it on your own if you plan to do so.
     *
     * This function returns the count of the observations examined, so to
     * produce the average log likelihood, just divide each element of the
     * derivative by this number.
     *
     * @param grid_ The data that will be used for this computation.
     * @param start_ The row to start at.
     * @param end_ The row to end at, half open interval (end_ is not included
     * in the computation).
     * @param regressorPointers_ An array of indices to the regressors you are
     * interested in (may not be the full R x S matrix you want).
     * @param statusPointers_ An array of indices to the statuses you are
     * interested in.
     * @param derivative_ The function's primary output, the derivative of the
     * LL, the i'th element of which is with respect to the
     * regressorPointer_[i]'th regressor and the statusPointers_[i]'th status.
     * @return The total observation count used for this computation.
     */
    public int computeDerivative(final ParamFittingGrid<S, R, T> grid_, final int start_, final int end_, final int[] regressorPointers_, final int[] statusPointers_, final double[] derivative_)
    {
        final int dimension = regressorPointers_.length;
        final double[] computed = _probWorkspace;
        final double[] actual = _actualProbWorkspace;
        final double[] regressors = _regWorkspace;
        final double[] rawReg = _rawRegWorkspace;
        final List<S> reachable = getParams().getStatus().getReachable();
        int count = 0;

        //For safety
        Arrays.fill(derivative_, 0.0);

        final int fromOrdinal = _params.getStatus().ordinal();

        for (int i = start_; i < end_; i++)
        {
            if (grid_.getStatus(i) != fromOrdinal)
            {
                continue;
            }
            if (!grid_.hasNextStatus(i))
            {
                continue;
            }

            //We inline a bunch of these calcs to reduce duplication of effort.
            grid_.getRegressors(i, rawReg);
            this.fillWorkspace(rawReg, regressors);
            rawPowerScores(regressors, computed);
            MultiLogistic.multiLogisticFunction(computed, computed);

            final int actualTransition = grid_.getNextStatus(i);

            final int actualOffset = _likelihood.ordinalToOffset(actualTransition);

            if (actualOffset < 0)
            {
                //This is a supposedly impossible transition, skip it.
                continue;
            }

            Arrays.fill(actual, 0.0);
            actual[actualOffset] = 1.0;

            for (int z = 0; z < dimension; z++)
            {
                //looping over the betas.
                double betaTerm = 0.0;

                for (int q = 0; q < reachable.size(); q++)
                {
                    //looping over the to-states.
                    final double betaDerivative = betaDerivative(regressors, computed, regressorPointers_[z], q, statusPointers_[z]);
                    final double actualProbability = actual[q];
                    final double computedProb = computed[q];
                    final double contribution = actualProbability * betaDerivative / computedProb;

                    if (Double.isNaN(contribution) || Double.isInfinite(contribution))
                    {
                        LOG.severe("Derivative term contribution is NaN or infinite, this should not be possible: " + contribution);
                        LOG.severe("Trying to recover, derivative may be somewhat inaccurate.");
                        break;
                    }

                    betaTerm += contribution;
                }

                derivative_[z] += betaTerm;
            }

            count++;
        }

        return count;
    }

    /**
     * N.B: This is NOT threadsafe. It contains some use of internal state that
     * cannot be shared. Clone the model as needed.
     *
     * @param grid_ The grid holding the data
     * @param index_ The row in the grid representing this data point
     * @param output_ The array to hold the output transition probabilities
     * @return The number of probabilities written into output_
     */
    public int transitionProbability(final ItemParamGrid<S, R, T> grid_, final int index_, final double[] output_)
    {
        grid_.getRegressors(index_, _rawRegWorkspace);
        return transitionProbability(_rawRegWorkspace, output_);
    }

    /**
     * Compute the transition probability from raw regressors. Be fairly careful
     * here, it's not easy to know which order the regressors need to come in.
     *
     * @param regs_ The raw regressor values (ordered to match
     * uniqueRegressors() from the ItemParams under this model)
     * @param output_ A workspace to hold the resulting transition
     * probabilities.
     * @return The number of transition probabilities.
     */
    public int transitionProbability(final double[] regs_, final double[] output_)
    {
        multiLogisticFunction(regs_, output_);
        return _betas.length;
    }

    public void powerScores(final double[] regressors_, final double[] workspace_)
    {
        if (regressors_.length != _rawRegWorkspace.length)
        {
            throw new IllegalArgumentException("Length mismatch: " + regressors_.length + " != " + _rawRegWorkspace.length);
        }
        if (workspace_.length != _betas.length)
        {
            throw new IllegalArgumentException("Length mismatch: " + workspace_.length + " != " + _reachableSize);
        }

        fillWorkspace(regressors_, _regWorkspace);
        rawPowerScores(_regWorkspace, workspace_);
    }

    private void rawPowerScores(final double[] regressors_, final double[] workspace_)
    {
        final int inputSize = _regWorkspace.length;
        final int outputSize = _betas.length;

        for (int i = 0; i < outputSize; i++)
        {
            double sum = 0.0;
            final double[] betaArray = _betas[i];

            for (int k = 0; k < inputSize; k++)
            {
                sum += regressors_[k] * betaArray[k];
            }

            workspace_[i] = sum;
        }
    }

    private void multiLogisticFunction(final double[] regressors_, final double[] workspace_)
    {
        powerScores(regressors_, workspace_);
        MultiLogistic.multiLogisticFunction(workspace_, workspace_);
    }

    public double betaDerivative(final double[] regressors_, final double[] computedProbabilities_, final int regressorIndex_, final int toStateIndex_, final int toStateBetaIndex_)
    {
        final double output = MultiLogistic.multiLogisticBetaDerivative(regressors_, computedProbabilities_, regressorIndex_, toStateIndex_, toStateBetaIndex_);
        return output;
    }

    /**
     * Note that this method is synchronized. Therefore, it may be called from
     * any thread, and you are guaranteed to get a viable model within that
     * thread.
     *
     * @return A clone of this model
     */
    @Override
    public final synchronized ItemModel<S, R, T> clone()
    {
        //Yes, yes, this is bad form. However, this class is final and I don't feel like
        //making all its internal variables not-final so that I can use the proper clone
        //idiom.
        return new ItemModel<>(this.getParams());
    }
}
