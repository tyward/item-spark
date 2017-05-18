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
package edu.columbia.tjw.item.util;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 *
 * @author tyler
 */
public final class LogUtil
{

//    static
//    {
//        final Logger rootLogger = Logger.getLogger("");
//
//        for (final Handler next : rootLogger.getHandlers())
//        {
//            rootLogger.removeHandler(next);
//        }
//
//        rootLogger.addHandler(new InnerHandler());
//    }
    private LogUtil()
    {
    }

    public static Logger getLogger(final Class<?> clazz_)
    {
        final String name = clazz_.getName();
        final Logger output = Logger.getLogger(name);
        return output;
    }

    private static final class InnerHandler extends Handler
    {
        private final DateFormat _format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

        @Override
        public synchronized void publish(LogRecord record)
        {
            //Let's format this thing nicely. 
            final Date recDate = new Date(record.getMillis());
            final String dateString = _format.format(recDate);
            final String message = record.getMessage();
            final String logger = record.getLoggerName();

            final StringBuffer builder = new StringBuffer();

            builder.append("[");
            builder.append(dateString);
            builder.append("][");
            builder.append(logger);
            builder.append("]: ");
            builder.append(message);

            final String completed = builder.toString();
            System.out.println(completed);

            final Throwable t = record.getThrown();

            if (null != t)
            {
                t.printStackTrace(System.out);
            }
        }

        @Override
        public void flush()
        {
            System.out.flush();
        }

        @Override
        public void close() throws SecurityException
        {
            //Do nothing.
        }

    }

}
