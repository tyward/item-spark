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

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 *
 * @author tyler
 */
public class GeneralThreadPool extends ThreadPoolExecutor
{
    private static final int NUM_PROCESSORS = Runtime.getRuntime().availableProcessors();
    private static final int MAX_THREADS = 32;
    private static final int BASE_SIZE = Math.min(NUM_PROCESSORS, MAX_THREADS);
    private static final int MAX_SIZE = Math.min(2 * NUM_PROCESSORS, MAX_THREADS);
    private static final GeneralThreadPool SINGLETON = new GeneralThreadPool();

    public static GeneralThreadPool singleton()
    {
        return SINGLETON;
    }

    public GeneralThreadPool(final int maxSize_)
    {
        super(Math.max(1, maxSize_ / 2), maxSize_, 500, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());
    }

    private GeneralThreadPool()
    {
        super(BASE_SIZE, MAX_SIZE, 500, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());

        this.setThreadFactory(new GeneralFactory());

    }

    private static final class GeneralFactory implements ThreadFactory
    {

        @Override
        public Thread newThread(Runnable r)
        {
            final Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }

    }

}
