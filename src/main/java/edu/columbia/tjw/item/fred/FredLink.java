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
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/**
 *
 * This class is designed to fetch data from the FRED XML API.
 *
 * You can find the API docs here: https://api.stlouisfed.org/docs/fred/
 *
 * FRED itself is here: https://research.stlouisfed.org/fred2/
 *
 * @author tyler
 */
public final class FredLink
{
    private static final int CLASS_HASH = HashUtil.startHash(FredLink.class);
    private static final String PROTOCOL = "https";
    private static final String HOST = "api.stlouisfed.org";
    private static final String SERIES_PATH = "/fred/series";
    private static final String CATEGORY_PATH = "/fred/category";
    private static final String CHILDREN_PATH = "/fred/category/children";
    private static final String CAT_SERIES_PATH = "/fred/category/series";
    private static final String OBSERVATION_PATH = "/fred/series/observations";

    private final Proxy _proxy;
    private final String _queryBase;
    private final String _apiKey;
    private final DocumentBuilder _builder;
    private final Map<String, FredSeries> _seriesMap;
    private final Map<String, FredException> _exceptionMap;
    private final Map<String, FredCategory> _categoryMap;
    private final Map<FredCategory, FredNavigationNode> _nodeMap;

    public FredLink(final String apiKey_)
    {
        this(apiKey_, null);
    }

    public FredLink(final String apiKey_, final Proxy proxy_)
    {
        _proxy = proxy_;
        _apiKey = apiKey_;
        _queryBase = "api_key=" + _apiKey + "&";

        final DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();

        dbf.setValidating(false);
        dbf.setIgnoringComments(false);
        dbf.setIgnoringElementContentWhitespace(true);
        dbf.setNamespaceAware(true);
        // dbf.setCoalescing(true);
        // dbf.setExpandEntityReferences(true);

        try
        {
            _builder = dbf.newDocumentBuilder();
            _builder.setEntityResolver(new NullResolver());
        }
        catch (final ParserConfigurationException e)
        {
            //Realistically, there's nothing the user could do about this anyway.
            throw new RuntimeException(e);
        }

        _seriesMap = new HashMap<>();
        _exceptionMap = new HashMap<>();
        _categoryMap = new HashMap<>();
        _nodeMap = new HashMap<>();
    }

    public synchronized FredNavigationNode getNode(final FredCategory cat_) throws IOException, FredException
    {
        final FredNavigationNode cached = _nodeMap.get(cat_);

        if (null != cached)
        {
            return cached;
        }

        final FredNavigationNode node = new FredNavigationNode(cat_);

        _nodeMap.put(cat_, node);

        return node;
    }

    private synchronized SortedMap<String, FredSeries> getChildSeries(final FredCategory cat_) throws IOException, FredException
    {
        final String catQuery = "category_id=" + cat_.getId();

        final Element catRoot = fetchData(CAT_SERIES_PATH, catQuery);
        final String rootName = catRoot.getTagName();

        if (!rootName.equals("seriess"))
        {
            throw new FredException("Unexpected tag name: " + rootName, 101);
        }

        final NodeList list = catRoot.getElementsByTagName("series");

        final SortedMap<String, FredSeries> nodes = new TreeMap<>();

        for (int i = 0; i < list.getLength(); i++)
        {
            final Element next = (Element) list.item(i);
            final String seriesName = next.getAttribute("id");
            final FredSeries series = this.fetchSeries(seriesName);
            nodes.put(series.getId(), series);
        }

        return nodes;
    }

    private synchronized SortedSet<FredNavigationNode> getChildren(final FredCategory cat_) throws IOException, FredException
    {
        final String catQuery = "category_id=" + cat_.getId();

        final Element catRoot = fetchData(CHILDREN_PATH, catQuery);
        final NodeList list = catRoot.getElementsByTagName("category");

        final SortedSet<FredNavigationNode> nodes = new TreeSet<>();

        for (int i = 0; i < list.getLength(); i++)
        {
            final Element next = (Element) list.item(i);
            FredCategory nextCat = new FredCategory(next);

            final String catKey = "category:" + nextCat.getId();

            if (!_categoryMap.containsKey(catKey))
            {
                _categoryMap.put(catKey, nextCat);
            }
            else
            {
                nextCat = _categoryMap.get(catKey);
            }

            final FredNavigationNode nextNode = getNode(nextCat);
            nodes.add(nextNode);
        }

        return nodes;
    }

    public synchronized FredCategory fetchCategory(final int categoryId_) throws IOException, FredException
    {
        final String catName = "category:" + categoryId_;

        if (_categoryMap.containsKey(catName))
        {
            final FredCategory cat = _categoryMap.get(catName);

            if (null == cat)
            {
                final FredException e = _exceptionMap.get(catName);
                throw new FredException(e);
            }

            return cat;
        }

        final String catQuery = "category_id=" + categoryId_;

        try
        {
            final Element catRoot = fetchData(CATEGORY_PATH, catQuery);
            final Element catElem = (Element) catRoot.getElementsByTagName("category").item(0);
            final FredCategory output = new FredCategory(catElem);

            _categoryMap.put(catName, output);

            return output;
        }
        catch (final FredException e)
        {
            _categoryMap.put(catName, null);
            _exceptionMap.put(catName, e);
            throw new FredException(e);
        }

    }

    public synchronized FredSeries fetchSeries(final String seriesName_) throws IOException, FredException
    {
        final String seriesName = "series:" + seriesName_;

        if (_seriesMap.containsKey(seriesName))
        {
            final FredSeries series = _seriesMap.get(seriesName);

            if (null == series)
            {
                final FredException e = _exceptionMap.get(seriesName);
                throw new FredException(e);
            }

            return series;
        }

        final String seriesQuery = "series_id=" + seriesName_;

        try
        {
            final Element seriesRoot = fetchData(SERIES_PATH, seriesQuery);
            final Element obsRoot = fetchData(OBSERVATION_PATH, seriesQuery);
            final FredSeriesData obs = new FredSeriesData(obsRoot);
            final Element seriesElem = (Element) seriesRoot.getElementsByTagName("series").item(0);
            final FredSeries output = new FredSeries(seriesElem, obs);

            _seriesMap.put(seriesName, output);

            return output;
        }
        catch (final FredException e)
        {
            _seriesMap.put(seriesName, null);
            _exceptionMap.put(seriesName, e);
            throw new FredException(e);
        }
    }

    private void checkError(final Element elem_) throws FredException
    {
        final String tagName = elem_.getTagName();

        if (!tagName.equals("error"))
        {
            return;
        }

        final String code = elem_.getAttribute("code");
        final String error = elem_.getAttribute("message");

        final int codeInt;

        if (null == code)
        {
            codeInt = -1;
        }
        else
        {
            codeInt = Integer.parseInt(code);
        }

        final FredException exc = new FredException(error, codeInt);
        throw exc;
    }

    private Element fetchData(final String pathName_, final String query_) throws IOException, FredException
    {
        final String fullQuery = pathName_ + "?" + _queryBase + query_;
        final URL thisUrl = new URL(PROTOCOL, HOST, fullQuery);

        final HttpURLConnection conn;

        if (null != _proxy)
        {
            conn = (HttpURLConnection) thisUrl.openConnection(_proxy);
        }
        else
        {
            conn = (HttpURLConnection) thisUrl.openConnection();
        }

        final int responseCode = conn.getResponseCode();

        if (400 == responseCode)
        {
            throw new FredException("Element does not exist: " + query_, responseCode);
        }

        try (final InputStream stream = conn.getInputStream())
        {
            final Document output = readXml(stream);
            final Element outputElem = output.getDocumentElement();

            this.checkError(outputElem);

            return outputElem;
        }
        catch (final SAXException e)
        {
            throw new IOException("XML exception.", e);
        }

    }

    public Document readXml(InputStream is) throws SAXException, IOException
    {
        return _builder.parse(is);
    }

    private static final class NullResolver implements EntityResolver
    {
        @Override
        public InputSource resolveEntity(String publicId, String systemId) throws SAXException,
                IOException
        {
            return new InputSource(new StringReader(""));
        }

    }

    /**
     * N.B: This class is not serializable as it requires the repeated use of
     * FRED.
     */
    public final class FredNavigationNode implements Comparable<FredNavigationNode>
    {

        private final FredCategory _category;
        private SortedSet<FredNavigationNode> _children;
        private SortedMap<String, FredSeries> _series;

        public FredNavigationNode(final FredCategory cat_)
        {
            _category = cat_;
            _children = null;
            _series = null;
        }

        public FredCategory getCategory()
        {
            return _category;
        }

        public synchronized SortedMap<String, FredSeries> getSeries() throws IOException, FredException
        {
            if (null != _series)
            {
                return _series;
            }

            _series = Collections.unmodifiableSortedMap(FredLink.this.getChildSeries(_category));
            return _series;
        }

        public synchronized SortedSet<FredNavigationNode> getChildren() throws IOException, FredException
        {
            if (null != _children)
            {
                return _children;
            }

            _children = FredLink.this.getChildren(_category);

            return _children;
        }

        @Override
        public int hashCode()
        {
            final int hash = HashUtil.mix(CLASS_HASH, _category.hashCode());
            return hash;
        }

        @Override
        public boolean equals(final Object that_)
        {
            if (this == that_)
            {
                return true;
            }
            if (null == that_)
            {
                return false;
            }
            if (this.getClass() != that_.getClass())
            {
                return false;
            }

            final FredNavigationNode that = (FredNavigationNode) that_;
            final int comp = this.compareTo(that);
            final boolean equal = (0 == comp);
            return equal;
        }

        @Override
        public int compareTo(FredNavigationNode that_)
        {
            if (null == that_)
            {
                return 1;
            }
            if (this == that_)
            {
                return 0;
            }

            return this._category.compareTo(that_.getCategory());
        }

    }

}
