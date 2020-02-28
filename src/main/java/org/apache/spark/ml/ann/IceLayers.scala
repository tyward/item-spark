package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, sum => Bsum}
import breeze.numerics.{log => brzlog}
import org.apache.spark.InternalAccumulator.input
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

private[ann] trait GeneralIceLayer extends Layer {

  override def createModel(initialWeights: BDV[Double]): GeneralIceLayerModel

}

/**
 * Trait for loss function
 */
private[ann] trait IceLossFunction {
  /**
   * Returns the value of loss function.
   * Computes loss based on target and output.
   * Writes delta (error) to delta in place.
   * Delta is allocated based on the outputSize
   * of model implementation.
   *
   * @param output actual output
   * @param target target output
   * @param delta delta (updated in place)
   * @return loss
   */
  def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double], gradRatio: BDM[Double]): Double
}

private[ann] trait GeneralIceLayerModel extends LayerModel {

  def computePrevDeltaExpanded(delta: BDM[Double], gradRatio: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit

  def grad2(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double], cumG2: BDM[Double]): Unit

}

/**
 * Functional layer properties, y = f(x)
 *
 */
private[ann] class IceSigmoidLayer extends GeneralIceLayer {

  val activationFunction: SigmoidFunctionIce = new SigmoidFunctionIce();

  override val weightSize = 0

  override def getOutputSize(inputSize: Int): Int = inputSize

  override val inPlace = true

  override def createModel(weights: BDV[Double]): GeneralIceLayerModel = new IceLayerLayerModel(this)

  override def initModel(weights: BDV[Double], random: Random): GeneralIceLayerModel =
    createModel(weights)
}

/**
 * Functional layer model. Holds no weights.
 *
 * @param layer functional layer
 */
private[ann] class IceLayerLayerModel private[ann](val layer: IceSigmoidLayer)
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

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {}

  override def grad2(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double], cumG2: BDM[Double]): Unit = {
    grad(delta, input, cumGrad)
  }

  override def computePrevDeltaExpanded(delta: BDM[Double], gradRatio: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);
    ApplyInPlace(output, gradRatio, layer.activationFunction.secondDerivativeRatio);
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

  def secondDerivativeRatio: (Double) => Double = z => {
    // this is f'' / f', which happens to have a really simple form.
    (1 - 2*z)
  }
}

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

  override def computePrevDeltaExpanded(delta: BDM[Double], gradRatio: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit = {
    computePrevDelta(delta, output, prevDelta);
  }

  override def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double], gradRatio: BDM[Double]): Double = {
    ApplyInPlace(output, target, delta, (o: Double, t: Double) => o - t)

    // Compute the gradient ratio, ugly but should work.
    var i = 0
    while (i < target.cols) {
      var j = 0
      while (j < target.rows) {
        val yi = output(j, i);
        val ti = target(j, i);
        var di = 0;

        if(ti == j) {
          di = 1;
        }

        val ratio = yi*(di - yi) / (yi - ti);
        gradRatio(j, i) = ratio

        j += 1
      }
      i += 1
    }



    -Bsum(target *:* brzlog(output)) / output.cols
  }
}

/**
 * Model of Feed Forward Neural Network.
 * Implements forward, gradient computation and can return weights in vector format.
 *
 * @param weights  network weights
 * @param topology network topology
 */
private[ml] class IceFeedForwardModel private(
                                               val weights: Vector,
                                               val topology: IceFeedForwardTopology) extends TopologyModel {
  val typedLayers: Array[GeneralIceLayer] = topology.layers
  val typedLayerModels = new Array[GeneralIceLayerModel](typedLayers.length)

  val layers = new Array[Layer](typedLayers.length);
  val layerModels = new Array[LayerModel](layers.length)

  private var offset = 0
  for (i <- 0 until typedLayers.length) {
    layers(i) = typedLayers(i);
    typedLayerModels(i) = typedLayers(i).createModel(
      new BDV[Double](weights.toArray, offset, 1, typedLayers(i).weightSize))
    layerModels(i) = typedLayerModels(i);
    offset += typedLayers(i).weightSize
  }
  private var outputs: Array[BDM[Double]] = null
  private var deltas: Array[BDM[Double]] = null
  private var gradRatio: Array[BDM[Double]] = null

  override def forward(data: BDM[Double], includeLastLayer: Boolean): Array[BDM[Double]] = {
    // Initialize output arrays for all layers. Special treatment for InPlace
    val currentBatchSize = data.cols
    // TODO: allocate outputs as one big array and then create BDMs from it
    if (outputs == null || outputs(0).cols != currentBatchSize) {
      outputs = new Array[BDM[Double]](layers.length)
      var inputSize = data.rows
      for (i <- 0 until layers.length) {
        if (layers(i).inPlace) {
          outputs(i) = outputs(i - 1)
        } else {
          val outputSize = layers(i).getOutputSize(inputSize)
          outputs(i) = new BDM[Double](outputSize, currentBatchSize)
          inputSize = outputSize
        }
      }
    }
    layerModels(0).eval(data, outputs(0))
    val end = if (includeLastLayer) layerModels.length else layerModels.length - 1
    for (i <- 1 until end) {
      layerModels(i).eval(outputs(i - 1), outputs(i))
    }
    outputs
  }

  override def computeGradient(
                                data: BDM[Double],
                                target: BDM[Double],
                                cumGradient: Vector,
                                realBatchSize: Int): Double = {
    val outputs = forward(data, true)
    val currentBatchSize = data.cols
    // TODO: allocate deltas as one big array and then create BDMs from it
    if (deltas == null || deltas(0).cols != currentBatchSize) {
      deltas = new Array[BDM[Double]](layerModels.length)
      gradRatio = new Array[BDM[Double]](layerModels.length)
      var inputSize = data.rows
      for (i <- 0 until layerModels.length - 1) {
        val outputSize = layers(i).getOutputSize(inputSize)
        deltas(i) = new BDM[Double](outputSize, currentBatchSize)
        gradRatio(i) = new BDM[Double](outputSize, currentBatchSize)
        inputSize = outputSize
      }
    }
    val L = layerModels.length - 1
    // TODO: explain why delta of top layer is null (because it might contain loss+layer)
    val loss = layerModels.last match {
      case levelWithError: IceLossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1), gradRatio(L - 1))
      case _ =>
        throw new UnsupportedOperationException("Top layer is required to have objective.")
    }
    for (i <- (L - 2) to(0, -1)) {
      typedLayerModels(i + 1).computePrevDeltaExpanded(deltas(i + 1), gradRatio(i), outputs(i + 1), deltas(i))
    }
    val cumGradientArray = cumGradient.toArray
    var offset = 0
    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      typedLayerModels(i).grad2(deltas(i), input,
        new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize), gradRatio(i))
      offset += layers(i).weightSize
    }
    loss
  }

  override def predict(data: Vector): Vector = {
    val size = data.size
    val result = forward(new BDM[Double](size, 1, data.toArray), true)
    Vectors.dense(result.last.toArray)
  }

  override def predictRaw(data: Vector): Vector = {
    val result = forward(new BDM[Double](data.size, 1, data.toArray), false)
    Vectors.dense(result(result.length - 2).toArray)
  }

  override def raw2ProbabilityInPlace(data: Vector): Vector = {
    val dataMatrix = new BDM[Double](data.size, 1, data.toArray)
    layerModels.last.eval(dataMatrix, dataMatrix)
    data
  }
}

/**
 * Fabric for feed forward ANN models
 */
private[ann] object IceFeedForwardModel {

  /**
   * Creates a model from a topology and weights
   *
   * @param topology topology
   * @param weights  weights
   * @return model
   */
  def apply(topology: IceFeedForwardTopology, weights: Vector): IceFeedForwardModel = {
    val expectedWeightSize = topology.layers.map(_.weightSize).sum
    require(weights.size == expectedWeightSize,
      s"Expected weight vector of size ${expectedWeightSize} but got size ${weights.size}.")
    new IceFeedForwardModel(weights, topology)
  }

  /**
   * Creates a model given a topology and seed
   *
   * @param topology topology
   * @param seed     seed for generating the weights
   * @return model
   */
  def apply(topology: IceFeedForwardTopology, seed: Long = 11L): IceFeedForwardModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    val weights = BDV.zeros[Double](topology.layers.map(_.weightSize).sum)
    var offset = 0
    val random = new XORShiftRandom(seed)
    for (i <- 0 until layers.length) {
      layerModels(i) = layers(i).
        initModel(new BDV[Double](weights.data, offset, 1, layers(i).weightSize), random)
      offset += layers(i).weightSize
    }
    new IceFeedForwardModel(Vectors.fromBreeze(weights), topology)
  }
}


/**
 * Feed forward ANN
 *
 * @param layers Array of layers
 */
private[ann] class IceFeedForwardTopology(val layers: Array[GeneralIceLayer]) extends Topology {
  override def model(weights: Vector): TopologyModel = IceFeedForwardModel(this, weights)

  override def model(seed: Long): TopologyModel = IceFeedForwardModel(this, seed)
}

/**
 * Factory for some of the frequently-used topologies
 */
private[ml] object IceFeedForwardTopology {

  /**
   * Creates a feed forward topology from the array of layers
   *
   * @param layers array of layers
   * @return feed forward topology
   */
  def apply(layers: Array[GeneralIceLayer]): IceFeedForwardTopology = {
    new IceFeedForwardTopology(layers)
  }


  /**
   * Creates a multi-layer perceptron
   *
   * @param layerSizes sizes of layers including input and output size
   * @return multilayer perceptron topology
   */
  def multiLayerPerceptron(
                            layerSizes: Array[Int]): IceFeedForwardTopology = {
    val layers = new Array[GeneralIceLayer]((layerSizes.length - 1) * 2)
    for (i <- 0 until layerSizes.length - 1) {
      layers(i * 2) = new IceAffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (i == layerSizes.length - 2) {
          new IceLossLayer()
        } else {
          new IceSigmoidLayer()
        }
    }
    IceFeedForwardTopology(layers)
  }
}




/**
 * Layer properties of affine transformations, that is y=A*x+b
 *
 * @param numIn number of inputs
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
 * @param layer layer properties
 */
private[ann] class IceAffineLayerModel private[ann] (
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
   * @param layer layer properties
   * @param weights vector for weights initialization
   * @param random random number generator
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
   * @param numIn number of inputs
   * @param numOut number of outputs
   * @param weights vector for weights initialization
   * @param random random number generator
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