package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV}
import edu.columbia.tjw.item.algo.DoubleVector
import edu.columbia.tjw.item.util.IceTools


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

  private var nextLayer: GeneralIceLayerModel = null

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

  override def singleGrad(delta: BDM[Double], input: BDM[Double], m: Int, weightGradB: BDM[Double], biasGradB: BDV[Double]): Unit = {
    var i = 0;
    while (i < weightGradB.rows) {
      val delta_i = delta(i, m);
      biasGradB(i) = delta_i;

      var k = 0;
      while (k < weightGradB.cols) {
        val a_k = input(k, m)
        weightGradB(i, k) = delta_i * a_k;
        k += 1;
      }

      i += 1;
    }
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

  override def grad2(delta: BDM[Double], nextDelta: BDM[Double], gamma: BDM[Double], input: BDM[Double], output: BDM[Double], cumG2: BDV[Double]): Unit = {
    val cumG2ofWeights = new BDM[Double](w.rows, w.cols, cumG2.data, cumG2.offset)
    val cumG2ofBias = new BDV[Double](cumG2.data, cumG2.offset + w.size, 1, b.length)

    var targetDelta: BDM[Double] = null;

    if (nextDelta != null) {
      targetDelta = nextDelta;
    }
    else {
      targetDelta = delta;
    }

    // Loop over
    var m = 0;
    while (m < gamma.cols) {

      var i = 0;
      while (i < cumG2ofWeights.rows) {
        // Compute scale factor.
        val gamma_i = gamma(i, m);
        val prevOutput_i = output(i, m);

        val fprime_i = nextLayer.activationDeriv(prevOutput_i);
        val fprime2_i = nextLayer.activationSecondDeriv(prevOutput_i)
        val delta_i = targetDelta(i, m);

        val scale = (gamma_i * fprime_i * fprime_i) + (delta_i * fprime2_i)
        cumG2ofBias(i) += scale;

        var k = 0;
        while (k < cumG2ofWeights.cols) {
          val a_k = input(k, m);
          val computedG2 = scale * a_k * a_k;

          cumG2ofWeights(i, k) += computedG2;
          k += 1;
        }

        i += 1;
      }

      m += 1;
    }

    val invObsCount = 1.0 / gamma.cols;

    var i = 0;
    while (i < cumG2.length) {
      cumG2(i) = cumG2(i) * invObsCount;
      i += 1;
    }
  }

  override def computePrevDeltaExpanded(delta: BDM[Double], nextDelta: BDM[Double], gamma: BDM[Double], prevOutput: BDM[Double], output: BDM[Double], prevDelta: BDM[Double], prevGamma: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);

    var targetDelta: BDM[Double] = null;
    var nextIsLoss = false;

    if (nextDelta != null) {
      targetDelta = nextDelta;
    }
    else {
      targetDelta = delta;
      nextIsLoss = true;
    }

    // Loop over observations.
    var m = 0;
    while (m < gamma.cols) {
      // Loop over output indices.
      var i = 0;
      while (i < gamma.rows) {
        val gamma_i = gamma(i, m);
        val prevOutput_i = prevOutput(i, m);

        val fprime_i = nextLayer.activationDeriv(prevOutput_i);
        val fprime2_i = nextLayer.activationSecondDeriv(prevOutput_i)
        val delta_i = targetDelta(i, m);

        val deltaTerm = delta_i * fprime2_i;
        val gammaTerm = (gamma_i * fprime_i * fprime_i);
        val scale = gammaTerm + deltaTerm

        if (!nextIsLoss) {
          for (k <- 0 until w.cols) {
            val currWeight = w(i, k)
            val computedGamma = scale * currWeight * currWeight;
            // Inner summation.
            prevGamma(k, m) += computedGamma
          }
        }
        else {
          // N.B: There are significant cross terms in the loss layer (due to the mutual exclusivity of the outputs), so we should not ignore them.
          // On the plus side, these cross terms are easy and cheap to compute, so just do that here, computing W^T*J*W exactly.

          var k = 0;
          while (k < w.cols) {
            val w_k = w(i, k)
            prevGamma(k, m) += deltaTerm * w_k * w_k;

            var subSum = 0.0;

            var u = 0;
            while (u < gamma.rows) {
              val w_u = w(u, k);
              val output_u = prevOutput(u, m);
              var kron = 0.0;

              if (u == i) {
                kron = 1.0;
              }

              val elem = (kron - output_u) * prevOutput_i * w_u;
              subSum += elem;
              u += 1;
            }

            prevGamma(k, m) += subSum * w_k;
            k += 1;
          }
        }

        i += 1;
      }

      m += 1;
    }
  }

  override def activationDeriv(input: Double): Double = {
    1.0
  }

  override def activationSecondDeriv(input: Double): Double = {
    0.0
  }

  def setNextLayer(nextLayer: GeneralIceLayerModel): Unit = {
    this.nextLayer = nextLayer
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