package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV}


/**
 * Layer properties of affine transformations, that is y=A*x+b
 *
 * @param numIn  number of inputs
 * @param numOut number of outputs
 */
private[ann] class IceAffineLayer(val numIn: Int, val numOut: Int) extends GeneralIceLayer {

  override val weightSize = numIn * numOut + numOut

  override def getOutputSize(inputSize: Int): Int = numOut

  override val inPlace = false

  override def createModel(weights: BDV[Double]): GeneralIceLayerModel = new IceAffineLayerModel(weights, this)

  override def initModel(weights: BDV[Double], random: Random): LayerModel =
    IceAffineLayerModel(this, weights, random)
}

/**
 * Model of Affine layer
 *
 * @param weights weights
 * @param layer   layer properties
 */
private[ann] class IceAffineLayerModel private[ann](
                                                     val weights: BDV[Double],
                                                     val layer: IceAffineLayer) extends GeneralIceLayerModel {
  val w = new BDM[Double](layer.numOut, layer.numIn, weights.data, weights.offset)
  val b =
    new BDV[Double](weights.data, weights.offset + (layer.numOut * layer.numIn), 1, layer.numOut)

  private var ones: BDV[Double] = null

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    output(::, *) := b
    BreezeUtil.dgemm(1.0, w, data, 1.0, output)
  }


  override def computePrevDelta(
                                 delta: BDM[Double],
                                 output: BDM[Double],
                                 prevDelta: BDM[Double]): Unit = {
    BreezeUtil.dgemm(1.0, w.t, delta, 0.0, prevDelta)
  }

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {
    // compute gradient of weights
    val cumGradientOfWeights = new BDM[Double](w.rows, w.cols, cumGrad.data, cumGrad.offset)
    BreezeUtil.dgemm(1.0 / input.cols, delta, input.t, 1.0, cumGradientOfWeights)
    if (ones == null || ones.length != delta.cols) ones = BDV.ones[Double](delta.cols)
    // compute gradient of bias
    val cumGradientOfBias = new BDV[Double](cumGrad.data, cumGrad.offset + w.size, 1, b.length)
    BreezeUtil.dgemv(1.0 / input.cols, delta, ones, 1.0, cumGradientOfBias)
  }

  override def grad2(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double], cumG2: BDM[Double]): Unit = {
    grad(delta, input, cumGrad)
  }

  override def computePrevDeltaExpanded(delta: BDM[Double], gradRatio: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);
    // Nothing special here....
  }
}

/**
 * Fabric for Affine layer models
 */
private[ann] object IceAffineLayerModel {

  /**
   * Creates a model of Affine layer
   *
   * @param layer   layer properties
   * @param weights vector for weights initialization
   * @param random  random number generator
   * @return model of Affine layer
   */
  def apply(layer: IceAffineLayer, weights: BDV[Double], random: Random): IceAffineLayerModel = {
    randomWeights(layer.numIn, layer.numOut, weights, random)
    new IceAffineLayerModel(weights, layer)
  }

  /**
   * Initialize weights randomly in the interval.
   * Uses [Bottou-88] heuristic [-a/sqrt(in); a/sqrt(in)],
   * where `a` is chosen in such a way that the weight variance corresponds
   * to the points to the maximal curvature of the activation function
   * (which is approximately 2.38 for a standard sigmoid).
   *
   * @param numIn   number of inputs
   * @param numOut  number of outputs
   * @param weights vector for weights initialization
   * @param random  random number generator
   */
  def randomWeights(
                     numIn: Int,
                     numOut: Int,
                     weights: BDV[Double],
                     random: Random): Unit = {
    var i = 0
    val sqrtIn = math.sqrt(numIn)
    while (i < weights.length) {
      weights(i) = (random.nextDouble * 4.8 - 2.4) / sqrtIn
      i += 1
    }
  }
}