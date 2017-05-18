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
package edu.columbia.tjw.item.optimize;

import edu.columbia.tjw.item.util.thread.GeneralTask;
import edu.columbia.tjw.item.util.thread.GeneralThreadPool;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author tyler
 */
public abstract class ThreadedMultivariateFunction implements MultivariateFunction
{
    private static final GeneralThreadPool POOL = GeneralThreadPool.singleton();
    private final int _blockSize;
    private final boolean _useThreading;
    private final Object _prepLock = new Object();

    public ThreadedMultivariateFunction(final int blockSize_, final boolean useThreading_)
    {
        _blockSize = blockSize_;
        _useThreading = useThreading_;
    }

    @Override
    public abstract int dimension();

    public abstract int resultSize(final int start_, final int end_);

    @Override
    public synchronized final void value(MultivariatePoint input_, int start_, int end_, EvaluationResult result_)
    {
        if (start_ == end_)
        {
            //do nothing.
            return;
        }
        if (start_ > end_)
        {
            throw new IllegalArgumentException("Start must be less than end.");
        }
        if (start_ < 0)
        {
            throw new IllegalArgumentException("Start must be nonnegative.");
        }

        synchronized (_prepLock)
        {
            prepare(input_);
        }

        final int numRows = (end_ - start_);
        final int numTasks = 1 + (numRows / _blockSize);

        final List<FunctionTask> taskList = new ArrayList<>(numTasks);

        for (int i = 0; i < numTasks; i++)
        {
            final int thisStart = start_ + (i * _blockSize);

            if (thisStart > end_)
            {
                break;
            }

            final int blockEnd = thisStart + _blockSize;
            final int thisEnd = Math.min(end_, blockEnd);

            if (thisEnd == thisStart)
            {
                break;
            }

            final int taskSize = this.resultSize(thisStart, thisEnd);
            final FunctionTask task = new FunctionTask(thisStart, thisEnd, taskSize);
            taskList.add(task);
        }

        final List<EvaluationResult> results = executeTasks(taskList);

        //Some synchronization to make sure we don't read old data.
        synchronized (result_)
        {
            for (final EvaluationResult next : results)
            {
                synchronized (next)
                {
                    result_.add(next, result_.getHighWater(), next.getHighRow());
                }
            }
        }
    }

    private <W> List<W> executeTasks(final List<? extends GeneralTask<W>> tasks_)
    {
        final List<W> output = new ArrayList<>(tasks_.size());

        for (final GeneralTask<W> next : tasks_)
        {
            if (_useThreading)
            {
                POOL.execute(next);
            }
            else
            {
                next.run();
            }
        }

        for (final GeneralTask<W> next : tasks_)
        {
            final W res = next.waitForCompletion();
            output.add(res);
        }

        return output;
    }

    public synchronized final MultivariateGradient calculateDerivative(MultivariatePoint input_, EvaluationResult result_, double precision_)
    {
        synchronized (_prepLock)
        {
            this.prepare(input_);
        }

        final int start = 0;
        final int end = this.numRows();

        final int numRows = (end - start);
        final int numTasks = 1 + (numRows / _blockSize);

        final List<DerivativeTask> taskList = new ArrayList<>(numTasks);
 
        for (int i = 0; i < numTasks; i++)
        {
            final int thisStart = start + (i * _blockSize);

            if (thisStart > end)
            {
                break;
            }

            final int blockEnd = thisStart + _blockSize;
            final int thisEnd = Math.min(end, blockEnd);

            if (thisEnd == thisStart)
            {
                break;
            }

            final int size = (thisEnd - thisStart);
            final DerivativeTask task = new DerivativeTask(thisStart, thisEnd, input_, result_, size);
            taskList.add(task);
        }

        final List<MultivariateGradient> results = executeTasks(taskList);

        int totalSize = 0;
        final double[] gradient = new double[results.get(0).getGradient().getDimension()];

        for (int i = 0; i < results.size(); i++)
        {
            final MultivariateGradient next = results.get(i);
            final DerivativeTask task = taskList.get(i);

            synchronized (task)
            {
                final int size = task.getRowCount();
                totalSize += size;

                final MultivariatePoint nextGrad = next.getGradient();

                for (int w = 0; w < nextGrad.getDimension(); w++)
                {
                    final double gradVal = nextGrad.getElement(w);
                    final double weighted = size * gradVal;
                    gradient[w] += weighted;
                }
            }
        }

        if (totalSize > 0)
        {
            for (int w = 0; w < gradient.length; w++)
            {
                gradient[w] = gradient[w] / totalSize;
            }
        }

        final MultivariatePoint point = new MultivariatePoint(gradient);

        final MultivariateGradient grad = new MultivariateGradient(input_, point, null, 0.0);
        return grad;
    }

    @Override
    public abstract int numRows();

    protected abstract void prepare(final MultivariatePoint input_);

    protected abstract void evaluate(final int start_, final int end_, EvaluationResult result_);

    protected abstract MultivariateGradient evaluateDerivative(final int start_, final int end_, MultivariatePoint input_, EvaluationResult result_);

    @Override
    public EvaluationResult generateResult(int start_, int end_)
    {
        final int resultSize = this.resultSize(start_, end_);
        final EvaluationResult output = new EvaluationResult(resultSize);
        return output;
    }

    @Override
    public EvaluationResult generateResult()
    {
        return generateResult(0, this.numRows());
    }

    private final class DerivativeTask extends GeneralTask<MultivariateGradient>
    {
        private final int _start;
        private final int _end;
        private final int _rowCount;
        private final MultivariatePoint _input;
        private final EvaluationResult _result;

        public DerivativeTask(final int start_, final int end_, MultivariatePoint input_, EvaluationResult result_, int rowCount_)
        {
            if (end_ <= start_)
            {
                throw new IllegalArgumentException("Invalid.");
            }
            _start = start_;
            _end = end_;
            _input = input_;
            _result = result_;
            _rowCount = rowCount_;
        }

        public int getRowCount()
        {
            return _rowCount;
        }

        @Override
        protected MultivariateGradient subRun()
        {
            final EvaluationResult res;

            synchronized (_prepLock)
            {
                res = _result;
            }

            synchronized (this)
            {
                return ThreadedMultivariateFunction.this.evaluateDerivative(_start, _end, _input, res);
            }
        }

    }

    private final class FunctionTask extends GeneralTask<EvaluationResult>
    {
        private final int _start;
        private final int _end;
        private final int _evalCount;

        public FunctionTask(final int start_, final int end_, final int evalCount_)
        {
            if (end_ <= start_)
            {
                throw new IllegalArgumentException("Invalid.");
            }
            _start = start_;
            _end = end_;
            _evalCount = evalCount_;
        }

        @Override
        protected EvaluationResult subRun()
        {
            final EvaluationResult res;

            synchronized (_prepLock)
            {
                res = new EvaluationResult(_evalCount);
            }

            synchronized (res)
            {
                ThreadedMultivariateFunction.this.evaluate(_start, _end, res);
            }

            return res;
        }

    }

}
