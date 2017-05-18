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
package edu.columbia.tjw.fred;

import java.io.Serializable;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.TreeMap;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/**
 *
 * @author tyler
 */
public final class FredSeriesData implements Serializable
{
    private static final long serialVersionUID = 7422230574929407270L;

    private final LocalDate[] _dates;
    private final double[] _values;

    protected FredSeriesData(final Element elem_)
    {
        final String tagName = elem_.getTagName();

        if (!tagName.equals("observations"))
        {
            throw new IllegalArgumentException("Invalid element: " + tagName);
        }

        final NodeList children = elem_.getElementsByTagName("observation");
        final int childCount = children.getLength();

        final Map<String, String> dateMap = new TreeMap<>();

        for (int i = 0; i < childCount; i++)
        {
            final Element next = (Element) children.item(i);

            final String dateString = next.getAttribute("date");
            final String valueString = next.getAttribute("value");

            if (valueString.equals("."))
            {
                //This is a special value used by FRED to indicate that the data is not available. 
                //Typically, this is because the markets are closed for some reason. Just skip it. 
                continue;
            }

            dateMap.put(dateString, valueString);
        }

        final int dataCount = dateMap.size();
        _dates = new LocalDate[dataCount];
        _values = new double[dataCount];

        int pointer = 0;

        for (final Map.Entry<String, String> entry : dateMap.entrySet())
        {
            final String dateString = entry.getKey();
            final String valueString = entry.getValue();

            _dates[pointer] = LocalDate.from(DateTimeFormatter.ISO_LOCAL_DATE.parse(dateString));
            _values[pointer] = Double.parseDouble(valueString);
            pointer++;
        }
    }

    public int size()
    {
        return _dates.length;
    }

    public double getValue(final int index_)
    {
        return _values[index_];
    }

    public LocalDate getDate(final int index_)
    {
        return _dates[index_];
    }

}
