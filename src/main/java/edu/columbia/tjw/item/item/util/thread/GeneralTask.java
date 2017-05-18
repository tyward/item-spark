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
package edu.columbia.tjw.item.util.thread;

/**
 *
 * @author tyler
 * @param <V> The return type of this task
 */
public abstract class GeneralTask<V> implements Runnable
{
    private final Object _runLock;
    private Throwable _exception;
    private V _result;
    private boolean _isDone;
    private boolean _isRunning;

    public GeneralTask()
    {
        _runLock = new Object();

        synchronized (_runLock)
        {
            //This is being used as a memory barrier. We are making sure that the
            //thread calling the constructor flushes all its info to main memory 
            //where it will be visible to any thread running this task (which will also hold the runLock). 
            _exception = null;
            _result = null;
            _isDone = false;
            _isRunning = false;
        }
    }

    public synchronized boolean isRunning()
    {
        return _isRunning;
    }

    public synchronized boolean isDone()
    {
        return _isDone;
    }

    public V waitForCompletion()
    {
        //If we are not done, and not already running, then let's run the task
        //rather than just waiting for it. Since run kicks out if it's already running or done,
        //it is safe to call this unconditionally.
        this.run();

        synchronized (this)
        {
            while (!_isDone)
            {
                try
                {
                    this.wait();
                }
                catch (final InterruptedException e)
                {
                    throw new RuntimeException(e);
                }
            }
        }

        synchronized (_runLock)
        {
            if (null != _exception)
            {
                throw new RuntimeException(_exception);
            }

            return _result;
        }
    }

    @Override
    public void run()
    {
        synchronized (this)
        {
            if (this.isRunning() || this.isDone())
            {
                return;
            }
            else
            {
                _isRunning = true;
            }
        }

        V result = null;
        Throwable t = null;

        synchronized (_runLock)
        {
            try
            {
                result = subRun();
                t = null;
            }
            catch (final Throwable t_)
            {
                t = t_;
                result = null;
            }
            finally
            {
                synchronized (this)
                {
                    this._result = result;
                    this._exception = t;
                    this._isDone = true;
                    this._isRunning = false;

                    this.notifyAll();
                }
            }
        }
    }

    protected abstract V subRun() throws Exception;
}
