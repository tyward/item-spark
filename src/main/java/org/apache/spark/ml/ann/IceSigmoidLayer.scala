package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import edu.columbia.tjw.item.algo.DoubleVector


/**
 * Functional layer properties, y = f(x)
 *
 */
private[ann] class IceSigmoidLayer extends GeneralIceLayer {

  val activationFunction: SigmoidFunctionIce = new SigmoidFunctionIce();

  override val weightSize = 0

  override def getOutputSize(inputSize: Int): Int = inputSize

  override val inPlace = true

  override def createModel(weights: BDV[Double]): GeneralIceLayerModel = new IceSigmoidLayerModel(this)

  override def initModel(weights: BDV[Double], random: Random): GeneralIceLayerModel =
    createModel(weights)
}

/**
 * Functional layer model. Holds no weights.
 *
 * @param layer functional layer
 */
private[ann] class IceSigmoidLayerModel private[ann](val layer: IceSigmoidLayer)
  extends GeneralIceLayerModel {

  // empty weights
  val weights = new BDV[Double](0)

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    ApplyInPlace(data, output, layer.activationFunction.eval)
  }

  override def computePrevDelta(
                                 nextDelta: BDM[Double],
                                 input: BDM[Double],
                                 delta: BDM[Double]): Unit = {
    ApplyInPlace(input, delta, layer.activationFunction.derivative)
    delta :*= nextDelta
  }

  override def singleGrad(delta: BDM[Double], input: BDM[Double], m: Int, weightGradB: BDM[Double], biasGradB: BDV[Double]): Unit = {}

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {}

  override def grad2(delta: BDM[Double], nextDelta: BDM[Double], gamma: BDM[Double], input: BDM[Double], output: BDM[Double], cumG2: BDV[Double]): Unit = {}

  override def computePrevDeltaExpanded(delta: BDM[Double], nextDelta: BDM[Double], gamma: BDM[Double], prevOutput: BDM[Double], output: BDM[Double], prevDelta: BDM[Double], prevGamma: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);

    // Copy gamma -> prevGamma
    for (i <- 0 until gamma.rows) {
      for (j <- 0 until gamma.cols) {
        prevGamma(i, j) = gamma(i, j)
      }
    }
  }

  override def activationDeriv(input: Double): Double = {
    layer.activationFunction.derivative(input)
  }

  override def activationSecondDeriv(input: Double): Double = {
    layer.activationFunction.secondDerivative(input)
  }

  def setNextLayer(nextLayer: GeneralIceLayerModel): Unit = {
    // Do nothing.
  }

}

/**
 * Implements Sigmoid activation function
 */
private[ann] class SigmoidFunctionIce extends ActivationFunction {

  override def eval: (Double) => Double = x => 1.0 / (1 + math.exp(-x))

  override def derivative: (Double) => Double = z => {

    (1 - z) * z
  }

  def secondDerivative: (Double) => Double = z => {
    // this is f'' / f', which happens to have a really simple form.
    val ratio = (1 - 2 * z);

    (ratio * derivative(z))
  }
}