/*
 * Copyright 2015 tyler.
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
 */
package edu.columbia.tjw.item.algo;

import edu.columbia.tjw.item.util.random.RandomTool;
import edu.columbia.tjw.item.algo.QuantApprox.QuantileNode;
import edu.columbia.tjw.item.util.LogUtil;
import edu.columbia.tjw.item.util.MathFunctions;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.logging.Logger;

/**
 *
 * This is an algorithm for approximating the quantiles of a distribution.
 *
 * Given a set of values (x, y), we are going to approximate the quantiles of x.
 *
 * More specifically, we are going to construct a set of N cutoffs, dividing the
 * space into N buckets, such that each bucket has an approximately equal number
 * of points.
 *
 * Then we are going to compute the approximate average and std. Dev of y over
 * each bucket.
 *
 * The goal is to construct an easy characterization of a distribution, which
 * can then be used for computations.
 *
 *
 * @author tyler
 */
public final class QuantApprox extends DistStats2D implements Iterable<QuantileNode>
{
    private static final Logger LOG = LogUtil.getLogger(QuantApprox.class);
    private static final long serialVersionUID = 8383047095582432490L;

    public static final int DEFAULT_LOAD = 10;
    public static final int DEFAULT_BUCKETS = 100;

    public static final int MIN_BUCKETS = 4;
    public static final int MIN_LOAD = 8;

    private final int _loadFactor;
    private final int _maxBuckets;

    private long _observationCount;
    private int _bucketCount;
    private QuantileNode _root;

    public static void main(final String[] args_)
    {
        try
        {
            final Random rand = RandomTool.getRandom();

            final int bucketCount = 100;
            final int loadFactor = 10;
            final int sampleSize = 100 * 1000;

            final QuantApprox approx = new QuantApprox(bucketCount, loadFactor);
            final SortedMap<Double, Double> valMap = new TreeMap<>();

            for (int i = 0; i < sampleSize; i++)
            {
                final double x = rand.nextGaussian();
                final double y = rand.nextGaussian() + x;

                approx.addObservation(x, y);
                valMap.put(x, y);
            }

            int index = 0;

            final Double[] allX = valMap.keySet().toArray(new Double[valMap.size()]);
            final Double[] allY = valMap.values().toArray(new Double[valMap.size()]);

            final double[] eX = new double[bucketCount];
            final double[] eY = new double[bucketCount];
            final double[] eY2 = new double[bucketCount];
            final double[] actStart = new double[bucketCount];
            final int blockSize = sampleSize / bucketCount;

            for (int i = 0; i < sampleSize; i++)
            {
                final int offset = i / blockSize;
                eX[offset] += allX[i];
                eY[offset] += allY[i];
                eY2[offset] += allY[i] * allY[i];
                actStart[offset] = allX[blockSize * offset];
            }

            for (int i = 0; i < bucketCount; i++)
            {
                eX[i] /= blockSize;
                eY[i] /= blockSize;
                eY2[i] /= blockSize;
                actStart[i] /= blockSize;
            }

            System.out.println("index, approxMin, approxEX, approxEY, approxDevY, approxCount, eX, eY, actStart, actDev");

            for (final QuantileNode next : approx)
            {

                System.out.println(index + ", " + next.getMinX() + ", " + next.getMeanX() + ", " + next.getMeanY() + ", " + next.getStdDevY() + ", " + next.getCount() + ", "
                        + eX[index] + ", " + eY[index] + ", " + actStart[index] + ", " + (eY2[index] - (eY[index] * eY[index])));
                index++;
            }

            System.out.println("Done.");

        }
        catch (final Exception e_)
        {
            e_.printStackTrace();
        }
    }

    public QuantApprox()
    {
        this(DEFAULT_BUCKETS, DEFAULT_LOAD);
    }

    public QuantApprox(final int maxBuckets_, final int loadFactor_)
    {
        if (maxBuckets_ < MIN_BUCKETS)
        {
            throw new IllegalArgumentException("Invalid bucket count.");
        }
        if (loadFactor_ < MIN_LOAD)
        {
            throw new IllegalArgumentException("Invalid load factor.");
        }

        _maxBuckets = maxBuckets_;
        _loadFactor = loadFactor_;
        _bucketCount = 0;
        _observationCount = 0;
    }

    public QuantileDistribution getDistribution()
    {
        return new QuantileDistribution(this);
    }

    public boolean isValidObservation(final double x_, final double y_)
    {
        final boolean xInvalid = Double.isNaN(x_) || Double.isInfinite(x_);
        final boolean yInvalid = Double.isNaN(y_) || Double.isInfinite(y_);
        final boolean invalid = xInvalid || yInvalid;
        return !invalid;
    }

    public void addObservation(final double x_, final double y_)
    {
        addObservation(x_, y_, true);
    }

    public void addObservation(final double x_, final double y_, final boolean skipInvalid_)
    {
        final boolean isValid = isValidObservation(x_, y_);

        if (!isValid)
        {
            if (skipInvalid_)
            {
                return;
            }

            throw new IllegalArgumentException("X and Y both need to be well defined numbers.");
        }

        if (null == _root)
        {
            _root = new QuantileNode(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, null);
        }

        _observationCount++;
        _root.addObservation(x_, y_);
        this.update(x_, y_);
    }

    public int size()
    {
        return _bucketCount;
    }

    public long observationCount()
    {
        return _observationCount;
    }

    @Override
    public Iterator<QuantileNode> iterator()
    {
        return new InnerIterator(_root);
    }

    private static final class InnerIterator implements Iterator<QuantileNode>
    {
        private QuantileNode _next;

        public InnerIterator(final QuantileNode node_)
        {
            _next = node_;
            seekLeft();
        }

        private void seekLeft()
        {
            while (null != _next.getChild(true))
            {
                _next = _next.getChild(true);
            }
        }

        @Override
        public boolean hasNext()
        {
            return null != _next;
        }

        @Override
        public QuantileNode next()
        {
            if (null == _next)
            {
                throw new NoSuchElementException();
            }

            final QuantileNode output = _next;

            if (null != _next.getChild(false))
            {
                _next = _next.getChild(false);
                seekLeft();
            }
            else
            {
                //Have exhausted this subtree, move up until we traverse a left link.
                QuantileNode current = output;
                _next = current._parent;

                while (null != _next && _next.getChild(false) == current)
                {
                    current = _next;
                    _next = current._parent;
                }

            }

            return output;
        }

    }

    public final class QuantileNode extends DistStats2D
    {
        private static final long serialVersionUID = 5520011448976824958L;

        //This node represents a half open interval. 
        private double _start;
        private final double _end;

        private double _xMax;
        private double _xMin;

        private int _height;
        private QuantileNode _leftChild;
        private QuantileNode _rightChild;
        private QuantileNode _parent;

        public QuantileNode(final double start_, final double end_, final QuantileNode parent_)
        {
            QuantApprox.this._bucketCount++;
            _start = start_;
            _end = end_;

            _parent = parent_;
            _leftChild = null;
            _rightChild = null;
            _height = 1;
        }

        private QuantileNode(final double start_, final double end_, final QuantileNode parent_, final double xMax_, final double xMin_)
        {
            this(start_, end_, parent_);

            _xMin = xMin_;
            _xMax = xMax_;
        }

        public double getMinX()
        {
            return _xMin;
        }

        public double getMaxX()
        {
            return _xMax;
        }

        private void addObservation(final double x_, final double y_)
        {
            //OK, this observation belongs to us....
            //First, see if we need to split. 
            final boolean splitProposed = (this.getCount() >= _loadFactor) && (_bucketCount < _maxBuckets);

            //N.B: We may have point masses, whereby some of the buckets can't be split. 
            //We can always identify those as the ones with zero variance, so don't try to split those or we will get empty buckets. 
            //It is possible (but not likely) to have empty buckets anyway, but we shouldn't make a point of having lots of them...
            if (splitProposed && this.getVarX() > 0.0)
            {
                //Do the split here. 
                //First, just make a new left child, this will involve some approximation. 
                final double eX = this.getMeanX();
//                final double eX2 = _x2Sum / _count;
//                final double varX = eX2 - (eX * eX);
//
//                //Let's approximate what the sums might look like. 
//                //Ideally, we would try to make the variance match, using similar logic. For now, let's not bother. 
//                final double leftX = _count * ((_xMin + eX) * 0.5);
//                final double leftX2 = _count * (((_xMin * _xMin) + eX2) * 0.5);
//                final double rightX = _count * ((eX + _xMax) * 0.5);
//                final double rightX2 = _count * ((eX2 + (_xMax * _xMax)) * 0.5);
//
//                final double newY = _ySum * 0.5;
//                final double newY2 = _y2Sum * 0.5;
//                final double newCount = _count * 0.5;
//
//                final int thisBalance = this.calculateBalanceFactor();

                //LOG.info("Exising balance factor: " + thisBalance);
                final QuantileNode newNode = new QuantileNode(_start, eX, this, eX, _xMin);

                _start = eX;
                _xMin = eX;
                this.reset();

                final QuantileNode rebalanced;

                //Now, insert the node as the rightmost node in the left subtree. 
                if (null == this._leftChild)
                {
                    this.setChild(newNode, true);
                    rebalanced = this;
                }
                else
                {
                    QuantileNode candidate = this._leftChild;

                    while (null != candidate.getChild(false))
                    {
                        candidate = candidate.getChild(false);
                    }

                    candidate.setChild(newNode, false);
                    rebalanced = candidate;
                }

                this.addObservation(x_, y_);
                rebalanced.rebalance();

                return;
            }

            if (x_ < _start)
            {
                if (null == this._leftChild)
                {
                    throw new NullPointerException();
                }

                //N.B: Left child will not be null in this case, because of the way the intervals are constructed. 
                _leftChild.addObservation(x_, y_);
                return;
            }
            if (x_ >= _end)
            {
                if (null == this._rightChild)
                {
                    throw new NullPointerException();
                }

                _rightChild.addObservation(x_, y_);
                return;
            }

            this.update(x_, y_);
            _xMax = Math.max(_xMax, x_);
            _xMin = Math.min(_xMin, x_);
        }

        private QuantileNode rebalance(final boolean left_)
        {
            final QuantileNode origParent = this._parent;

            //Convert left-right case into left-left case (if applicable)
            final QuantileNode base = getChild(left_);
            final QuantileNode rotated = base.rotate(left_);
            this.setChild(rotated, left_);

            //Now we are guaranteed to have left-left (or right-right)
            //Finnish the rotation. 
            final QuantileNode newRoot = this.rotate(!left_);

            if (null != origParent)
            {
                final boolean isLeft = origParent.isLeftChild(this);
                origParent.setChild(newRoot, isLeft);
            }
            else
            {
                QuantApprox.this._root = newRoot;
                newRoot.setParent(null);
            }

            return newRoot;
        }

        private void rebalance()
        {
            this.fillHeight();
            final int balanceFactor = calculateBalanceFactor();
            final QuantileNode newRoot;

            switch (balanceFactor)
            {
                case 2:
                    newRoot = rebalance(true);
                    break;
                case -2:
                    newRoot = rebalance(false);
                    break;
                case 0:
                case -1:
                case 1:

                    newRoot = this;
                    //Do nothing. 
                    break;
                default:
                    throw new IllegalStateException("Not possible.");
            }

            final int newFactor = calculateBalanceFactor();
            //LOG.info("Rebalanced: " + balanceFactor + " -> " + newFactor);

            if (Math.abs(newFactor) > 1)
            {
                throw new IllegalStateException("Not possible.");
            }

            final QuantileNode newParent = newRoot._parent;

            if (null != newParent)
            {
                newParent.rebalance();
            }
        }

        private QuantileNode rotate(final boolean rotateLeft_)
        {
            //Rotate left will increase the balance factor by two, right will decrease it by two. 
            final int balanceFactor = calculateBalanceFactor();
            if (rotateLeft_ && balanceFactor > 0)
            {
                //Do nothing, we don't want to increase the balance factor. 
                return this;
            }
            if (!rotateLeft_ && balanceFactor < 0)
            {
                return this;
            }

            //We need to rotate, so let's do it. 
            final QuantileNode rotateChild = getChild(!rotateLeft_);
            final QuantileNode newChild = rotateChild.getChild(rotateLeft_);
            setChild(newChild, !rotateLeft_);
            rotateChild.setChild(this, rotateLeft_);
            this.fillHeight();
            rotateChild.fillHeight();
            return rotateChild;
        }

        private QuantileNode getChild(final boolean isLeftChild_)
        {
            if (isLeftChild_)
            {
                return _leftChild;
            }
            else
            {
                return _rightChild;
            }
        }

        private boolean isLeftChild(final QuantileNode child_)
        {
            if (_leftChild == child_)
            {
                return true;
            }
            else if (_rightChild == child_)
            {
                return false;
            }
            else
            {
                throw new IllegalStateException("Impossible.");
            }
        }

        private void setChild(final QuantileNode child_, final boolean isLeftChild_)
        {
            if (isLeftChild_)
            {
                _leftChild = child_;
            }
            else
            {
                _rightChild = child_;
            }

            if (null != child_)
            {
                child_.setParent(this);
            }
        }

        private void fillHeight()
        {
            _height = 1 + Math.max(calculateHeight(_leftChild), calculateHeight(_rightChild));
        }

        private void setParent(final QuantileNode parent_)
        {
            _parent = parent_;
        }

        public int getHeight()
        {
            return _height;
        }

        private int calculateBalanceFactor()
        {
            final int balanceFactor = calculateHeight(_leftChild) - calculateHeight(_rightChild);
            return balanceFactor;
        }

        private int calculateHeight(final QuantileNode target_)
        {
            if (null == target_)
            {
                return 0;
            }
            else
            {
                return target_.getHeight();
            }
        }

    }

}
