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
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveFactory;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.util.EnumFamily;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

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
    private final SimpleRegressor _intercept;
    private final ItemCurveFactory<SimpleRegressor, StandardCurveType> _factory;
    private final int _maxParamCount;
    private final List<SimpleRegressor> _regressors;
    private final SortedSet<SimpleRegressor> _curveRegressors;
    private final Set<SimpleRegressor> _nonCurveRegressors;


    /**
     * Use this to construct settings unless you really know what you're doing.
     *
     * @param settings_        Item settings to use, can be null.
     * @param intercept_       Which regressor will be the intercept. You don't need
     *                         to specify this in the data, it will be assumed to be 1.0 always.
     * @param status_          Which status are we projecting from (only interesting for
     *                         markov chains, just make a simple 2 status family if in doubt).
     * @param maxParamCount_   How many parameters is the model allowed to use.
     *                         Each curve will use approximately 3 params.
     * @param regressors_      The list (in order) of the regressors to use. This
     *                         must be in the same order that they appear in the features column.
     * @param curveRegressors_ The regressors that can support curves (i.e. they
     *                         are not binary flags), order doesn't matter for these.
     */
    public ItemClassifierSettings(final ItemSettings settings_, final String intercept_,
                                  final SimpleStatus status_, final int maxParamCount_, final List<String> regressors_, final Set<String> curveRegressors_)
    {
        if (null == settings_)
        {
            _settings = new ItemSettings();
        } else
        {
            _settings = settings_;
        }

        // First, make the regressor family.
        EnumFamily<SimpleRegressor> regFamily = SimpleRegressor.generateFamily(regressors_);

        _factory = new StandardCurveFactory<>();
        _fromStatus = status_;
        _intercept = regFamily.getFromName(intercept_);
        _maxParamCount = maxParamCount_;

        _regressors = Collections.unmodifiableList(new ArrayList<>(regFamily.getMembers()));

        SortedSet<SimpleRegressor> curveSet = new TreeSet<>();

        for (final String next : curveRegressors_)
        {
            curveSet.add(regFamily.getFromName(next));
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

        if ((_nonCurveRegressors.size() + _curveRegressors.size() + 1) != _regressors.size())
        {
            // Either _regressors has some repeated items, or curveRegressors contains items that are not in the regressors list.
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

    public SimpleRegressor getIntercept()
    {
        return _intercept;
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

}
