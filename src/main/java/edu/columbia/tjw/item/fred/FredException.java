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

/**
 *
 * @author tyler
 */
public final class FredException extends Exception
{
    private static final long serialVersionUID = 4594757215459405632L;

    private final int _code;

    public FredException(final FredException exc_)
    {
        super(exc_);

        _code = exc_.getCode();
    }

    public FredException(final String message_, final int code_)
    {
        super(message_);

        _code = code_;
    }

    public int getCode()
    {
        return _code;
    }

}
