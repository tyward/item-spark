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
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveFactory;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.util.EnumFamily;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * All the collected settings needed in order to run the ItemClassifier.
 *
 * @author tyler
 */
public final class ItemClassifierSettings implements Serializable
{
    private static final long serialVersionUID = 0xc6964a7d0bbd3449L;

    public ItemSettings _settings;
    private final SimpleStatus _fromStatus;

    @Deprecated
    private final SimpleRegressor _intercept;

    private final ItemCurveFactory<SimpleRegressor, StandardCurveType> _factory;
    private final int _maxParamCount;
    private final List<SimpleRegressor> _regressors;
    private final SortedSet<SimpleRegressor> _curveRegressors;
    private final Set<SimpleRegressor> _nonCurveRegressors;

    private final EnumFamily<SimpleRegressor> _regFamily;

    /**
     * Convenience constructor to allow easy changes to maxParamCount_
     *
     * @param base_
     * @param maxParamCount_
     */
    public ItemClassifierSettings(final ItemClassifierSettings base_, final int maxParamCount_) {
        _settings = base_._settings;
        _fromStatus = base_._fromStatus;
        _intercept = null;
        _factory = base_._factory;
        _maxParamCount = maxParamCount_;
        _regressors = base_._regressors;
        _curveRegressors = base_._curveRegressors;
        _nonCurveRegressors = base_._nonCurveRegressors;

        // N.B: Added to allow loading of previous version of ItemClassifierSettings
        if (base_._regFamily == null)
        {
            _regFamily = base_._intercept.getFamily();
        }
        else
        {
            _regFamily = base_._regFamily;
        }
    }


    public ItemClassifierSettings(final ItemClassifierSettings base_, final ItemSettings settings_)
    {
        _settings = settings_;
        _fromStatus = base_._fromStatus;
        _intercept = null;
        _factory = base_._factory;
        _maxParamCount = base_._maxParamCount;
        _regressors = base_._regressors;
        _curveRegressors = base_._curveRegressors;
        _nonCurveRegressors = base_._nonCurveRegressors;

        // N.B: Added to allow loading of previous version of ItemClassifierSettings
        if (base_._regFamily == null)
        {
            _regFamily = base_._intercept.getFamily();
        }
        else
        {
            _regFamily = base_._regFamily;
        }
    }

    /**
     * Use this to construct settings unless you really know what you're doing.
     *
     * @param settings_        Item settings to use, can be null.
     * @param status_          Which status are we projecting from (only interesting for
     *                         markov chains, just make a simple 2 status family if in doubt).
     * @param maxParamCount_   How many parameters is the model allowed to use.
     *                         Each curve will use approximately 3 params.
     * @param regressors_      The list (in order) of the regressors to use. This
     *                         must be in the same order that they appear in the features column.
     * @param curveRegressors_ The regressors that can support curves (i.e. they
     *                         are not binary flags), order doesn't matter for these.
     */
    public ItemClassifierSettings(final ItemSettings settings_,
                                  final SimpleStatus status_, final int maxParamCount_, final List<String> regressors_,
                                  final Set<String> curveRegressors_)
    {
        if (null == settings_)
        {
            _settings = new ItemSettings();
        }
        else
        {
            _settings = settings_;
        }

        // First, make the regressor family.
        _regFamily = SimpleRegressor.generateFamily(regressors_);

        _factory = new StandardCurveFactory<>();
        _fromStatus = status_;
        _intercept = null;
        _maxParamCount = maxParamCount_;

        _regressors = Collections.unmodifiableList(new ArrayList<>(_regFamily.getMembers()));

        SortedSet<SimpleRegressor> curveSet = new TreeSet<>();

        for (final String next : curveRegressors_)
        {
            curveSet.add(_regFamily.getFromName(next));
        }
        _curveRegressors = Collections.unmodifiableSortedSet(curveSet);

        final SortedSet<SimpleRegressor> nonCurveRegressors = new TreeSet<>();

        for (final SimpleRegressor next : _regressors)
        {
            if (!_curveRegressors.contains(next) && _intercept != next)
            {
                nonCurveRegressors.add(next);
            }
        }

        _nonCurveRegressors = Collections.unmodifiableSortedSet(nonCurveRegressors);

        if ((_nonCurveRegressors.size() + _curveRegressors.size()) != _regressors.size())
        {
            // Either _regressors has some repeated items, or curveRegressors contains items that are not in the
            // regressors list.
            throw new IllegalArgumentException("Regressor mismatch: Either regressors "
                    + "contains repeated items, or curveRegressors contains items not in regressors: "
                    + _nonCurveRegressors.toString() +
                    " \n: " + _curveRegressors.toString()
                    + " \n: " + _regressors.toString());
        }

    }

    public ItemSettings getSettings()
    {
        return _settings;
    }

    public SimpleStatus getFromStatus()
    {
        return _fromStatus;
    }

    public EnumFamily<SimpleRegressor> getRegressorFamily()
    {
        return _regFamily;
    }

    public ItemCurveFactory<SimpleRegressor, StandardCurveType> getFactory()
    {
        return _factory;
    }

    public int getMaxParamCount()
    {
        return _maxParamCount;
    }

    public List<SimpleRegressor> getRegressors()
    {
        return _regressors;
    }

    public SortedSet<SimpleRegressor> getCurveRegressors()
    {
        return _curveRegressors;
    }

    public Set<SimpleRegressor> getNonCurveRegressors()
    {
        return _nonCurveRegressors;
    }

    public void save(final String fileName_) throws IOException
    {
        try (final FileOutputStream fout = new FileOutputStream(fileName_);
             final GZIPOutputStream zipOut = new GZIPOutputStream(fout);
             final ObjectOutputStream oOut = new ObjectOutputStream(zipOut);)
        {
            oOut.writeObject(this);
        }
    }


    public static ItemClassifierSettings load(final String filename_) throws IOException
    {
        try (final FileInputStream fIn = new FileInputStream(filename_);
             final GZIPInputStream zipIn = new GZIPInputStream(fIn);
             final ObjectInputStream oIn = new ObjectInputStream(zipIn))
        {
            final ItemClassifierSettings settings = (ItemClassifierSettings) oIn.readObject();

            // N.B: Correcting for change in serialized form.
            if (settings._regFamily == null)
            {
                return new ItemClassifierSettings(settings, settings.getSettings());
            }

            return settings;
        }
        catch (ClassNotFoundException e)
        {
            throw new IOException("Unable to load unknown class.", e);
        }
    }

}
