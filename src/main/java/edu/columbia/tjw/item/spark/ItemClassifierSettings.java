/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemCurveFactory;
import edu.columbia.tjw.item.ItemCurveType;
import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemSettings;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.base.StandardCurveFactory;
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
 * @param <S>
 * @param <R>
 * @param <T>
 */
public final class ItemClassifierSettings<S extends ItemStatus<S>, R extends ItemRegressor<R>, T extends ItemCurveType<T>> implements Serializable
{
    private static final long serialVersionUID = 0xc6964a7d0bbd3449L;

    public ItemSettings _settings;
    private final S _fromStatus;
    private final R _intercept;
    private final ItemCurveFactory<R, T> _factory;
    private final int _maxParamCount;
    private final List<R> _regressors;
    private final SortedSet<R> _curveRegressors;
    private final Set<R> _nonCurveRegressors;

    /**
     * Use this constructor unless you really know what you're doing.
     *
     * @param intercept_ Which regressor will be the intercept. You don't need
     * to specify this in the data, it will be assumed to be 1.0 always.
     * @param status_ Which status are we projecting from (only interesting for
     * markov chains, just make a simple 2 status family if in doubt).
     * @param maxParamCount_ How many parameters is the model allowed to use.
     * Each curve will use approximately 3 params.
     * @param regressors_ The list (in order) of the regressors to use. This
     * must be in the same order that they appear in the features column.
     * @param curveRegressors_ The regressors that can support curves (i.e. they
     * are not binary flags), order doesn't matter for these.
     */
    public ItemClassifierSettings(final R intercept_, final S status_,
            final int maxParamCount_, final List<R> regressors_, final Set<R> curveRegressors_)
    {
        this(null, new StandardCurveFactory<>(), intercept_, status_, maxParamCount_, regressors_, curveRegressors_);
    }

    /**
     * See above.
     */
    public ItemClassifierSettings(final ItemSettings settings_, final ItemCurveFactory<R, T> factory_, final R intercept_,
            final S status_, final int maxParamCount_, final List<R> regressors_, final Set<R> curveRegressors_)
    {
        if (null == settings_)
        {
            _settings = new ItemSettings();
        }
        else
        {
            _settings = settings_;
        }

        _factory = factory_;
        _fromStatus = status_;
        _intercept = intercept_;
        _maxParamCount = maxParamCount_;

        _regressors = Collections.unmodifiableList(new ArrayList<>(regressors_));
        _curveRegressors = Collections.unmodifiableSortedSet(new TreeSet<>(curveRegressors_));

        final SortedSet<R> nonCurveRegressors = new TreeSet<>();

        for (final R next : _regressors)
        {
            if (!curveRegressors_.contains(next))
            {
                nonCurveRegressors.add(next);
            }
        }

        _nonCurveRegressors = Collections.unmodifiableSortedSet(nonCurveRegressors);

        if ((_nonCurveRegressors.size() + _curveRegressors.size()) != _regressors.size())
        {
            // Either _regressors has some repeated items, or curveRegressors contains items that are not in the regressors list.
            throw new IllegalArgumentException("Regressor mismatch: Either regressors "
                    + "contains repeated items, or curveRegressors contains items not in regressors.");
        }

    }

    public ItemSettings getSettings()
    {
        return _settings;
    }

    public S getFromStatus()
    {
        return _fromStatus;
    }

    public R getIntercept()
    {
        return _intercept;
    }

    public ItemCurveFactory<R, T> getFactory()
    {
        return _factory;
    }

    public int getMaxParamCount()
    {
        return _maxParamCount;
    }

    public List<R> getRegressors()
    {
        return _regressors;
    }

    public SortedSet<R> getCurveRegressors()
    {
        return _curveRegressors;
    }

    public Set<R> getNonCurveRegressors()
    {
        return _nonCurveRegressors;
    }

}
