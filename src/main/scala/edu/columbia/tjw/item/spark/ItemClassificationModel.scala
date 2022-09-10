package edu.columbia.tjw.item.spark

import edu.columbia.tjw.item.base.{SimpleRegressor, SimpleStatus, StandardCurveType}
import edu.columbia.tjw.item.fit.FitResult
import edu.columbia.tjw.item.util.random.RandomTool
import edu.columbia.tjw.item.{ItemModel, ItemParameters}
import org.apache.spark.ml.classification.{LogisticRegressionModel, ProbabilisticClassificationModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWriter}

import java.io._
import java.util


@SerialVersionUID(0x2bb4508735311c26L)
object ItemClassificationModel extends MLReadable[ItemClassificationModel] {
  def read: MLReader[ItemClassificationModel] = new ModelReaderSerializable[ItemClassificationModel]();

  @throws[IOException]
  override def load(filename_ : String):
  ItemClassificationModel = {
    var model = super.load(filename_)

    // Some properties don't restore correctly, rebuild this model to correct them.
    var output = new ItemClassificationModel(model._fitResult, model._settings)
    return output
  }
}

@SerialVersionUID(0x2bb4508735311c26L)
class ItemClassificationModel(override val uid: String, val _fitResult: FitResult[SimpleStatus, SimpleRegressor, StandardCurveType],
                              val _settings: ItemClassifierSettings)
  extends ProbabilisticClassificationModel[Vector, ItemClassificationModel]
    with org.apache.spark.ml.util.MLWritable
    with java.io.Serializable {

  def this(fitResult: FitResult[SimpleStatus, SimpleRegressor, StandardCurveType],
           settings: ItemClassifierSettings) {
    this(RandomTool.randomString(16), fitResult, settings);
  }

  def write: MLWriter = new ModelWriterSerializable[ItemClassificationModel](this);

  val paramFields: util.List[SimpleRegressor] = getParams.getUniqueRegressors
  final private var _offsetMap: Array[Int] = {
    var offsetMap = new Array[Int](paramFields.size)

    for (i <- 1 until paramFields.size) {
      val next: SimpleRegressor = paramFields.get(i)
      val index: Int = _settings.getRegressors.indexOf(next)
      if (-1 == index) throw new IllegalArgumentException("Missing regressors from fields.")
      offsetMap(i) = index
    }

    offsetMap
  }

  private var _model: ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType] = null
  private var _rawRegressors: Array[Double] = null

  def getSettings: ItemClassifierSettings = _settings

  override def raw2probabilityInPlace(rawProbabilities_ : Vector):
  Vector = { //Do nothing, these are already probabilities.
    return rawProbabilities_
  }

  override def numClasses: Int = getParams.getStatus.getReachableCount

  override def predictRaw(allRegressors_ : Vector):
  Vector = {
    val model: ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType] = getModel
    for (i <- 0 until getParams.getUniqueRegressors.size) {
      val fieldIndex: Int = _offsetMap(i)
      _rawRegressors(i) = allRegressors_.apply(fieldIndex)
    }
    val probabilities: Array[Double] = new Array[Double](getParams.getStatus.getReachableCount)
    model.transitionProbability(_rawRegressors, probabilities)
    return new DenseVector(probabilities)
  }

  override def copy(arg0: ParamMap): ItemClassificationModel = defaultCopy(arg0)

  def getFitResult: FitResult[SimpleStatus, SimpleRegressor, StandardCurveType] = _fitResult

  final def getParams: ItemParameters[SimpleStatus, SimpleRegressor, StandardCurveType] = getFitResult.getParams

  private def getModel: ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType] = {
    if (null == _model) {
      _model = new ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType](getParams)
      _rawRegressors = new Array[Double](getParams.getUniqueRegressors.size)
    }
    _model
  }
}