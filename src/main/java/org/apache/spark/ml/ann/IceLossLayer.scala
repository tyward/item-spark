package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, sum => Bsum}
import breeze.numerics.{log => brzlog}

private[ann] class IceLossLayer extends GeneralIceLayer {
  override val weightSize = 0
  override val inPlace = true

  override def getOutputSize(inputSize: Int): Int = inputSize

  override def createModel(weights: BDV[Double]): GeneralIceLayerModel =
    new IceCrossEntropyLossLayerModel()

  override def initModel(weights: BDV[Double], random: Random): GeneralIceLayerModel =
    new IceCrossEntropyLossLayerModel()
}


private[ann] class IceCrossEntropyLossLayerModel extends GeneralIceLayerModel with IceLossFunction {

  // loss layer models do not have weights
  val weights = new BDV[Double](0)

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    var j = 0
    // find max value to make sure later that exponent is computable
    while (j < data.cols) {
      var i = 0
      var max = Double.MinValue
      while (i < data.rows) {
        if (data(i, j) > max) {
          max = data(i, j)
        }
        i += 1
      }
      var sum = 0.0
      i = 0
      while (i < data.rows) {
        val res = math.exp(data(i, j) - max)
        output(i, j) = res
        sum += res
        i += 1
      }
      i = 0
      while (i < data.rows) {
        output(i, j) /= sum
        i += 1
      }
      j += 1
    }
  }

  override def computePrevDelta(
                                 nextDelta: BDM[Double],
                                 input: BDM[Double],
                                 delta: BDM[Double]): Unit = {
    /* loss layer model computes delta in loss function */
  }

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {
    /* loss layer model does not have weights */
  }

  override def grad2(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double], cumG2: BDM[Double]): Unit = {
    grad(delta, input, cumGrad)
  }

  override def computePrevDeltaExpanded(delta: BDM[Double], gamma: BDM[Double], output: BDM[Double], prevDelta: BDM[Double], prevGamma: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);
  }

  override def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double], gamma: BDM[Double]): Double = {
    ApplyInPlace(output, target, delta, (o: Double, t: Double) => o - t)

    var i = 0;
    while(i < target.cols) {
      var j = 0;
      var dot : Double = 0.0;

      while(j < target.rows) {
        var t = target(j, i);
        var o = output(j, i);

        dot += target(j, i) * output(j, i);
        j += 1;
      }

      j = 0;

      while(j < target.rows) {
        gamma(j, i) = (target(j, i) - dot) * output(j, i);
        j+= 1;
      }


      i += 1;
    }

    -Bsum(target *:* brzlog(output)) / output.cols
  }

  override def setNextWeights(weights: BDV[Double]): Unit = {
    // Do nothing, this will not be called.
  }
}