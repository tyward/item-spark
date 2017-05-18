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

import edu.columbia.tjw.item.util.HashUtil;
import java.io.Serializable;
import java.time.Instant;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import org.w3c.dom.Element;

/**
 *
 * @author tyler
 */
public final class FredSeries implements Serializable, Comparable<FredSeries>
{
    private static final long serialVersionUID = 364329225987396428L;
    private static final String FORMAT_PATTERN = "yyyy-MM-dd HH:mm:ssX";
    private static final int CLASS_HASH = HashUtil.startHash(FredSeries.class);

    private final String _id;
    private final String _title;
    private final String _units;
    private final String _notes;

    private final LocalDate _realtimeStart;
    private final LocalDate _realtimeEnd;
    private final LocalDate _observationStart;
    private final LocalDate _observationEnd;
    private final Instant _lastUpdated;

    private final String _frequency;
    private final String _seasonalAdjustment;

    private final int _popularity;
    private final FredSeriesData _series;
    private final int _hash;

    protected FredSeries(final Element elem_, final FredSeriesData series_)
    {
        final String tagName = elem_.getTagName();

        if (!tagName.equals("series"))
        {
            throw new IllegalArgumentException("Invalid element: " + tagName);
        }

        _id = elem_.getAttribute("id");
        _title = elem_.getAttribute("title");
        _units = elem_.getAttribute("units");
        _notes = elem_.getAttribute("notes");

        _realtimeStart = extractDate("realtime_start", elem_);
        _realtimeEnd = extractDate("realtime_end", elem_);
        _observationStart = extractDate("observation_start", elem_);
        _observationEnd = extractDate("observation_end", elem_);
        _lastUpdated = extractDateTime("last_updated", elem_);

        _frequency = elem_.getAttribute("frequency_short");
        _seasonalAdjustment = elem_.getAttribute("seasonal_adjustment_short");

        final String popString = elem_.getAttribute("popularity");

        if (null == popString)
        {
            _popularity = -1;
        }
        else
        {
            _popularity = Integer.parseInt(popString);
        }

        _series = series_;

        int hash = HashUtil.mix(CLASS_HASH, _id.hashCode());
        hash = HashUtil.mix(hash, _realtimeStart.hashCode());
        hash = HashUtil.mix(hash, _realtimeEnd.hashCode());
        hash = HashUtil.mix(hash, _observationStart.hashCode());
        hash = HashUtil.mix(hash, _observationEnd.hashCode());
        _hash = HashUtil.mix(hash, _lastUpdated.hashCode());
    }

    public String getId()
    {
        return _id;
    }

    public String getTitle()
    {
        return _title;
    }

    public String getUnits()
    {
        return _units;
    }

    public String getNotes()
    {
        return _notes;
    }

    public LocalDate getRealtimeStart()
    {
        return _realtimeStart;
    }

    public LocalDate getRealtimeEnd()
    {
        return _realtimeEnd;
    }

    public LocalDate getObservationStart()
    {
        return _observationStart;
    }

    public LocalDate getObservationEnd()
    {
        return _observationEnd;
    }

    public Instant getLastUpdated()
    {
        return _lastUpdated;
    }

    public String getFrequency()
    {
        return _frequency;
    }

    public String getSeasonalAdjustment()
    {
        return _seasonalAdjustment;
    }

    public int getPopularity()
    {
        return _popularity;
    }

    public FredSeriesData getSeries()
    {
        return _series;
    }

    private static LocalDate extractDate(final String attributeName_, final Element elem_)
    {
        final String val = elem_.getAttribute(attributeName_);

        if (null == val)
        {
            return null;
        }

        final LocalDate converted = LocalDate.from(DateTimeFormatter.ISO_LOCAL_DATE.parse(val));
        return converted;
    }

    private static Instant extractDateTime(final String attributeName_, final Element elem_)
    {
        final String val = elem_.getAttribute(attributeName_);

        if (null == val)
        {
            return null;
        }

        final DateTimeFormatter formatter = DateTimeFormatter.ofPattern(FORMAT_PATTERN);
        final TemporalAccessor t = formatter.parse(val);
        final OffsetDateTime dt = OffsetDateTime.from(t);

        final Instant converted = dt.toInstant();
        //final Instant converted = null;

        return converted;
    }

    @Override
    public int hashCode()
    {
        return _hash;
    }

    @Override
    public boolean equals(final Object other_)
    {
        if (null == other_)
        {
            return false;
        }
        if (this.getClass() != other_.getClass())
        {
            return false;
        }

        final FredSeries that = (FredSeries) other_;
        final int comparison = this.compareTo(that);
        return (0 == comparison);
    }

    @Override
    public int compareTo(FredSeries that_)
    {
        if (this == that_)
        {
            return 0;
        }
        if (null == that_)
        {
            return 1;
        }

        int comparison = this._id.compareTo(that_._id);

        if (comparison != 0)
        {
            return comparison;
        }

        //This would be very rare, same ID, but different object. Probably different dates, but which ones? 
        comparison = this._realtimeStart.compareTo(that_._realtimeStart);

        if (comparison != 0)
        {
            return comparison;
        }

        comparison = this._realtimeEnd.compareTo(that_._realtimeEnd);

        if (comparison != 0)
        {
            return comparison;
        }

        comparison = this._observationStart.compareTo(that_._observationStart);

        if (comparison != 0)
        {
            return comparison;
        }

        comparison = this._observationEnd.compareTo(that_._observationEnd);

        if (comparison != 0)
        {
            return comparison;
        }

        comparison = this._lastUpdated.compareTo(that_._lastUpdated);

        if (comparison != 0)
        {
            return comparison;
        }

        return 0;
    }

}
