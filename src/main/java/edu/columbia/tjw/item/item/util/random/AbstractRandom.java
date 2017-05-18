/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.columbia.tjw.item.util.random;

import edu.columbia.tjw.item.util.ByteTool;
import java.util.Random;

/**
 *
 * @author tyler
 */
public abstract class AbstractRandom extends Random
{
    public AbstractRandom()
    {
    }

    @Override
    protected final int next(int bits_)
    {
        if ((bits_ < 1) || (bits_ > 32))
        {
            throw new IllegalArgumentException("Invalid bit count: " + bits_);
        }

        final int raw = nextInt();
        final int output = raw >>> (32 - bits_);
        return output;
    }

    @Override
    public abstract int nextInt();

    public abstract void setSeed(final byte[] seed_);

    @Override
    public void setSeed(long seed_)
    {
        this.setSeed(ByteTool.longToBytes(seed_));
    }

    //Later on, add the ziggurat method for gaussians.
    @Override
    synchronized public double nextGaussian()
    {
        return super.nextGaussian();
    }
}
