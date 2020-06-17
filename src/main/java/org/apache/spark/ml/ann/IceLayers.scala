package org.apache.spark.ml.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import edu.columbia.tjw.item.algo.DoubleVector
import edu.columbia.tjw.item.util.IceTools
import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
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
   * @param delta  delta (updated in place)
   * @return loss
   */
  def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double], gammas: BDM[Double]): Double
}

private[ann] trait GeneralIceLayerModel extends LayerModel {

  def computePrevDeltaExpanded(delta: BDM[Double], gamma: BDM[Double], prevOutput: BDM[Double], output: BDM[Double], prevDelta: BDM[Double], prevGamma: BDM[Double]): Unit

  def gradIce(delta: BDM[Double], input: BDM[Double], g2: BDV[Double], g2Weight: BDV[Double], cumGrad: BDV[Double]): Double

  def grad2(delta: BDM[Double], gamma: BDM[Double], input: BDM[Double], output: BDM[Double], cumG2: BDV[Double]): Unit

  /**
   * Derivative of activation function, as a function of the output of the activation function.
   *
   * i.e. compute f'(x) as a function of f(x)
   *
   * @param input The value of the activation function. (i.e. f(x))
   * @return The derivative of the activation function (f'(x))
   */
  def activationDeriv(input: Double): Double

  /**
   * Same as above, expressed in terms of f(x), not x.
   *
   * @param input
   * @return
   */
  def activationSecondDeriv(input: Double): Double

  def setNextLayer(nextLayer: GeneralIceLayerModel): Unit

  //def setNextWeights(weights: BDV[Double]): Unit
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
  private var currWeights = new BDV[Double](weights.toArray, offset, 1, 0)

  for (i <- 0 until typedLayers.length) {
    layers(i) = typedLayers(i);

    val weightSize = typedLayers(i).weightSize;

    if (weightSize > 0) {
      currWeights = new BDV[Double](weights.toArray, offset, 1, typedLayers(i).weightSize)
    }

    typedLayerModels(i) = typedLayers(i).createModel(currWeights)

    if (i > 0) {
      typedLayerModels(i - 1).setNextLayer(typedLayerModels(i));
    }

    //    if (weightSize > 0 && i > 0) {
    //      typedLayerModels(i - 1).setNextWeights(currWeights);
    //    }

    layerModels(i) = typedLayerModels(i);
    offset += weightSize
  }
  private var outputs: Array[BDM[Double]] = null
  private var deltas: Array[BDM[Double]] = null
  private var gammas: Array[BDM[Double]] = null

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

    val cumGradientArray = cumGradient.toArray
    val cumG2Array = cumGradientArray.clone();

    return computeGradientRaw(data, target, cumGradient.toArray, cumG2Array);
  }


  def computeGradientRaw(
                          data: BDM[Double],
                          target: BDM[Double],
                          cumGradientArray: Array[Double],
                          cumG2Array: Array[Double]): Double = {
    val outputs = forward(data, true)
    val currentBatchSize = data.cols
    // TODO: allocate deltas as one big array and then create BDMs from it
    if (deltas == null || deltas(0).cols != currentBatchSize) {
      deltas = new Array[BDM[Double]](layerModels.length)
      gammas = new Array[BDM[Double]](layerModels.length)
      var inputSize = data.rows
      for (i <- 0 until layerModels.length - 1) {
        val outputSize = layers(i).getOutputSize(inputSize)
        deltas(i) = new BDM[Double](outputSize, currentBatchSize)
        gammas(i) = new BDM[Double](outputSize, currentBatchSize)
        inputSize = outputSize
      }
    }
    val L = layerModels.length - 1
    // TODO: explain why delta of top layer is null (because it might contain loss+layer)
    val loss = layerModels.last match {
      case levelWithError: IceLossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1), gammas(L - 1))
      case _ =>
        throw new UnsupportedOperationException("Top layer is required to have objective.")
    }
    for (i <- (L - 2) to(0, -1)) {
      typedLayerModels(i + 1).computePrevDeltaExpanded(deltas(i + 1), gammas(i + 1), outputs(i + 2), outputs(i + 1), deltas(i), gammas(i))
    }
    //val cumGradientArray = cumGradient.toArray
    //    val cumG2Array = cumGradientArray.clone();

    var offset = 0
    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      val g2Vec = new BDV[Double](cumG2Array, offset, 1, layers(i).weightSize)

      typedLayerModels(i).grad2(deltas(i), gammas(i), input, outputs(i), g2Vec)

      offset += layers(i).weightSize
    }

    //compute the weights...
    val g2Weights = IceTools.computeJWeight(DoubleVector.of(cumG2Array, false)).copyOfUnderlying();

    offset = 0;
    var lossAdj = 0.0;

    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      val gradVec = new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize)
      val g2Vec = new BDV[Double](cumG2Array, offset, 1, layers(i).weightSize)
      val g2Weight = new BDV[Double](g2Weights, offset, 1, layers(i).weightSize)

      lossAdj += typedLayerModels(i).gradIce(deltas(i), input, g2Vec, g2Weight, gradVec);

      offset += layers(i).weightSize
    }

    val iceLoss = loss + lossAdj;
    iceLoss
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
  override def model(weights: Vector): IceFeedForwardModel = IceFeedForwardModel(this, weights)

  override def model(seed: Long): IceFeedForwardModel = IceFeedForwardModel(this, seed)
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
      val affineLayer = new IceAffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2) = affineLayer
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
 * Neural network gradient. Does nothing but calling Model's gradient
 *
 * @param topology    topology
 * @param dataStacker data stacker
 */
private[ann] class ANNGradient2(topology: IceFeedForwardTopology, dataStacker: DataStacker) extends Gradient {
  def compute2(
                data: OldVector,
                label: Double,
                weights: OldVector,
                cumGradient: OldVector, cumG2Array: Array[Double]): Double = {
    val (input, target, realBatchSize) = dataStacker.unstack(Vectors.dense(data.toArray))
    val model: IceFeedForwardModel = topology.model(Vectors.dense(weights.toArray))

    val cumGradientArray = cumGradient.toArray
    //    val cumG2Array = cumGradientArray.clone();

    return model.computeGradientRaw(input, target, cumGradientArray, cumG2Array);

    //model.computeGradient(input, target, Vectors.dense(cumGradient.toArray), realBatchSize)
  }

  override def compute(
                        data: OldVector,
                        label: Double,
                        weights: OldVector,
                        cumGradient: OldVector): Double = {
    val cumGradientArray = cumGradient.toArray
    val cumG2Array = cumGradientArray.clone();

    return compute2(data, label, weights, cumGradient, cumG2Array);
    //
    //    val (input, target, realBatchSize) = dataStacker.unstack(Vectors.dense(data.toArray))
    //    val model: IceFeedForwardModel = topology.model(Vectors.dense(weights.toArray))
    //
    //    val cumGradientArray = cumGradient.toArray
    //    val cumG2Array = cumGradientArray.clone();
    //
    //    return model.computeGradientRaw(input, target, cumGradient.toArray, cumG2Array);

    //model.computeGradient(input, target, Vectors.dense(cumGradient.toArray), realBatchSize)
  }


}

class IcePerceptronClassificationModel private[ml](
                                                    @Since("1.5.0") override val uid: String,
                                                    @Since("1.5.0") override val layers: Array[Int],
                                                    @Since("2.0.0") override val weights: Vector, val blockSize: Int)
  extends MultilayerPerceptronClassificationModel(uid, layers, weights) {

  @Since("1.6.0")
  override val numFeatures: Int = layers.head

  def computeGradients(
                        dataset: Dataset[_],
                        weights: Vector,
                        cumGradientArray: Array[Double],
                        cumG2Array: Array[Double]): Double = {
    val myLayers: Array[Int] = layers
    val labels = myLayers.last
    val encodedLabelCol = "_encoded" + $(labelCol)
    val encodeModel = new OneHotEncoderModel(uid, Array(labels))
      .setInputCols(Array($(labelCol)))
      .setOutputCols(Array(encodedLabelCol))
      .setDropLast(false)
    val encodedDataset = encodeModel.transform(dataset)
    val data = encodedDataset.select($(featuresCol), encodedLabelCol).rdd.map {
      case Row(features: Vector, encodedLabel: Vector) => (features, encodedLabel)
    }

    val dataStacker = new DataStacker(blockSize, myLayers(0), myLayers.last)

    val iceModel: IceFeedForwardModel = IceFeedForwardTopology
      .multiLayerPerceptron(layers)
      .model(weights)

    val gradient: ANNGradient2 = new ANNGradient2(iceModel.topology, dataStacker)

    val trainData: RDD[(Double, OldVector)] = dataStacker.stack(data).map { v =>
      (v._1, OldVectors.fromML(v._2))
    }

    trainData.persist(StorageLevel.MEMORY_AND_DISK)

    val handlePersistence = (trainData.getStorageLevel == StorageLevel.NONE);

    try {
      if (handlePersistence) {
        trainData.persist(StorageLevel.MEMORY_AND_DISK)
      }

      // Do the work here...
      val w = OldVectors.dense(weights.toArray); //OldVectors.fromBreeze(weights)
      val n = w.size
      val bcW = data.context.broadcast(w)

      val seqOp = (c: (OldVector, OldVector, Double), v: (Double, OldVector)) =>
        (c, v) match {
          case ((grad, grad2, loss), (label, features)) =>
            val denseGrad: OldVector = grad.toDense
            val denseGrad2: OldVector = grad2.toDense
            val bcwV: OldVector = bcW.value
            val l = gradient.compute2(features, label, bcwV, denseGrad, denseGrad2.toArray)
            (denseGrad, denseGrad2, loss + l)
        }

      val combOp = (c1: (OldVector, OldVector, Double), c2: (OldVector, OldVector, Double)) =>
        (c1, c2) match {
          case ((grad1, diag1, loss1), (grad2, diag2, loss2)) =>
            val denseGrad1 = grad1.toDense
            val denseGrad2 = grad2.toDense
            axpy(1.0, denseGrad2, denseGrad1)

            val denseDiag1 = diag1.toDense;
            val denseDiag2 = diag2.toDense;
            axpy(1.0, denseDiag2, denseDiag1);

            (denseGrad1, denseDiag1, loss1 + loss2)
        }

      val zeroSparseVector = OldVectors.sparse(n, Seq.empty)
      val (gradientSum, gradient2Sum, lossSum) = trainData.treeAggregate((zeroSparseVector, zeroSparseVector, 0.0))(seqOp, combOp)

      for (w <- 0 until cumGradientArray.length) {
        cumGradientArray(w) = gradientSum(w);
        cumG2Array(w) = gradient2Sum(w);
      }

      return lossSum;
    }
    finally {
      if (handlePersistence) {
        trainData.unpersist()
      }
    }
  }

}

