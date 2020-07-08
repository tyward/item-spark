package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.ann.{FeedForwardTrainer, IceFeedForwardTopology, IcePerceptronClassificationModel}
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.{Dataset, Row}


/**
 * Classifier trainer based on the Multilayer Perceptron.
 * Each layer has sigmoid activation function, output layer has softmax.
 * Number of inputs has to be equal to the size of feature vectors.
 * Number of outputs has to be equal to the total number of labels.
 *
 */
@Since("1.5.0")
class IcePerceptronClassifier @Since("1.5.0")(
                                               @Since("1.5.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, IcePerceptronClassifier,
    MultilayerPerceptronClassificationModel]
    with MultilayerPerceptronParams with DefaultParamsWritable {

  final val iceScale: DoubleParam = new DoubleParam(this,
    "iceScale", "Ice effect scale, in [0, 1] where 0 is MLE and 1.0 is full ICE.",
    ParamValidators.inRange(0.0, 1.0));

  def setIceFactor(value: Double): this.type = set(iceScale, value)

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("mlpc"))

  /**
   * Sets the value of param [[layers]].
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /**
   * Sets the value of param [[blockSize]].
   * Default is 128.
   *
   * @group expertSetParam
   */
  @Since("1.5.0")
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /**
   * Sets the value of param [[solver]].
   * Default is "l-bfgs".
   *
   * @group expertSetParam
   */
  @Since("2.0.0")
  def setSolver(value: String): this.type = set(solver, value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setTol(value: Double): this.type = set(tol, value)

  /**
   * Set the seed for weights initialization if weights are not set
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[initialWeights]].
   *
   * @group expertSetParam
   */
  @Since("2.0.0")
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
   * Sets the value of param [[stepSize]] (applicable only for solver "gd").
   * Default is 0.03.
   *
   * @group setParam
   */
  @Since("2.0.0")
  def setStepSize(value: Double): this.type = set(stepSize, value)

  @Since("1.5.0")
  override def copy(extra: ParamMap): IcePerceptronClassifier = defaultCopy(extra)

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of `fit()` to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset Training dataset
   * @return Fitted model
   */
  override protected def train(
                                dataset: Dataset[_]): IcePerceptronClassificationModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, predictionCol, layers, maxIter, tol,
      blockSize, solver, stepSize, seed)

    val iceMult = this.get(iceScale).getOrElse(1.0);

    val myLayers = $(layers)
    val labels = myLayers.last
    instr.logNumClasses(labels)
    instr.logNumFeatures(myLayers.head)

    // One-hot encoding for labels using OneHotEncoderModel.
    // As we already know the length of encoding, we skip fitting and directly create
    // the model.
    val encodedLabelCol = "_encoded" + $(labelCol)
    val encodeModel = new OneHotEncoderModel(uid, Array(labels))
      .setInputCols(Array($(labelCol)))
      .setOutputCols(Array(encodedLabelCol))
      .setDropLast(false)
    val encodedDataset = encodeModel.transform(dataset)
    val data = encodedDataset.select($(featuresCol), encodedLabelCol).rdd.map {
      case Row(features: Vector, encodedLabel: Vector) => (features, encodedLabel)
    }

    val topology = IceFeedForwardTopology.multiLayerPerceptron(myLayers, dataset.count(), iceMult)
    val trainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }
    if ($(solver) == IcePerceptronClassifier.LBFGS) {
      trainer.LBFGSOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    } else if ($(solver) == IcePerceptronClassifier.GD) {
      trainer.SGDOptimizer
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by MultilayerPerceptronClassifier.")
    }
    trainer.setStackSize($(blockSize))
    val mlpModel = trainer.train(data)
    return new IcePerceptronClassificationModel(uid, myLayers, mlpModel.weights, $(blockSize))
  }
}


@Since("2.0.0")
object IcePerceptronClassifier
  extends DefaultParamsReadable[IcePerceptronClassifier] {

  /** String name for "l-bfgs" solver. */
  private[classification] val LBFGS = "l-bfgs"

  /** String name for "gd" (minibatch gradient descent) solver. */
  private[classification] val GD = "gd"

  /** Set of solvers that MultilayerPerceptronClassifier supports. */
  private[classification] val supportedSolvers = Array(LBFGS, GD)

  @Since("2.0.0")
  override def load(path: String): IcePerceptronClassifier = super.load(path)
}






