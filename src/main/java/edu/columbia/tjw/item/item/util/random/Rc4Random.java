/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.columbia.tjw.item.util.random;

import edu.columbia.tjw.item.util.HashTool;

/**
 *
 * @author tyler
 */
public final class Rc4Random extends AbstractRandom
{
    private static final long serialVersionUID = 5499153123879554L;
    private final byte[] _seed = new byte[32];
    private final byte[] _state = new byte[256];
    int _j = 0;
    int _i = 0;

    public Rc4Random(final byte[] input_)
    {
        super();

        for (int i = 0; i < _state.length; i++)
        {
            _state[i] = (byte) i;
        }

        this.setSeed(input_);
    }

    @Override
    public final int nextInt()
    {
        int next = nextByte() & 0xFF;
        int accumulator = next;

        accumulator = accumulator << 8;
        next = nextByte() & 0xFF;
        accumulator += next;

        accumulator = accumulator << 8;
        next = nextByte() & 0xFF;
        accumulator += next;

        accumulator = accumulator << 8;
        next = nextByte() & 0xFF;
        accumulator += next;

        return accumulator;
    }

    public final byte nextByte()
    {
        _i = (_i + 1) & 0xFF;
        _j = (_j + _state[_i]) & 0xFF;

        final byte temp = _state[_i];
        _state[_j] = _state[_i];
        _state[_i] = temp;

        final int counter = (_state[_i] + _state[_j]) & 0xFF;
        final byte output = _state[counter];
        return output;
    }

    @Override
    public void setSeed(byte[] seed_)
    {
        //Safely mix the new material, our state, and the previous material together.
        final byte[] whitened = HashTool.hash(seed_);
        final byte[] stateHash = HashTool.hash(_state);

        //Mix this into our seed pool in a way that is safe.
        for (int i = 0; i < _seed.length; i++)
        {
            _seed[i] = (byte) (_seed[i] + whitened[i]);
            _seed[i] = (byte) (_seed[i] ^ stateHash[i]);
        }

        //Now update the state to reflect the new material.
        for (int i = 0; i < _state.length; i++)
        {
            _state[i] = (byte) i;
        }

        _j = 0;
        _i = 0;

        for (int i = 0; i < _state.length; i++)
        {
            _j = _j + _state[i] + _seed[i % _seed.length];
            _j = _j & 0xFF;

            final byte temp = _state[i];
            _state[i] = _state[_j];
            _state[_j] = temp;
        }

        //Make sure that the value of _i is uniformly random.
        final int genCount = _seed[0] & 0xFF;

        for (int i = 0; i < genCount; i++)
        {
            nextByte();
        }
    }

}
