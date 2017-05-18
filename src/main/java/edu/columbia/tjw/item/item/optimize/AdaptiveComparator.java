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
package edu.columbia.tjw.item.optimize;

/**
 *
 * @author tyler
 * @param <V> The type of points on which this can be evaluated
 * @param <F> The type of optimization function which will be called
 */
public interface AdaptiveComparator<V extends EvaluationPoint<V>, F extends OptimizationFunction<V>>
{
    /**
     * Return (aRes - bRes) as a zScore.
     *
     * @param function_ The function to evaluate
     * @param a_ Point a on which to evaluate the function
     * @param b_ Point b on which to evaluate the function
     * @param aResult_ The result of evaluating function_(a_)
     * @param bResult_ THe result of evaluating function_(b_)
     * @return The difference between these two points, as a z-score
     */
    public double compare(final F function_, final V a_, final V b_, final EvaluationResult aResult_, final EvaluationResult bResult_);

    public double getSigmaTarget();

}
