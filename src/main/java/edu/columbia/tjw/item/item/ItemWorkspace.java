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
package edu.columbia.tjw.item;

/**
 * A workspace appropriate for use with ItemStatus S.
 *
 * The size of some of these vectors depends on the number of elements in the
 * family represented by the status type S.
 *
 * @author tyler
 * @param <S> The status type for this workspace.
 */
public final class ItemWorkspace<S extends ItemStatus<S>>
{

    private final double[] _regWorkspace;
    private final double[] _probWorkspace;
    private final double[] _actualProbWorkspace;

    public ItemWorkspace(final S status_, final int regressorCount_)
    {
        _regWorkspace = new double[regressorCount_];
        _probWorkspace = new double[status_.getReachable().size()];
        _actualProbWorkspace = new double[status_.getReachable().size()];
    }

    public double[] getRegressorWorkspace()
    {
        return _regWorkspace;
    }

    public double[] getComputedProbabilityWorkspace()
    {
        return _probWorkspace;
    }

    public double[] getActualProbabilityWorkspace()
    {
        return _actualProbWorkspace;
    }

}
