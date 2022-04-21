package edu.columbia.tjw.item.spark

import edu.columbia.tjw.item.base.{SimpleRegressor, SimpleStatus, StandardCurveType}
import edu.columbia.tjw.item.fit.FitResult
import edu.columbia.tjw.item.util.random.RandomTool
import edu.columbia.tjw.item.{ItemModel, ItemParameters}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.ParamMap

import java.io._
import java.util
import java.util.zip.{GZIPInputStream, GZIPOutputStream}


@SerialVersionUID(0x2bb4508735311c26L)
object ItemClassificationModel {
  @throws[IOException]
  def load(filename_ : String):
  ItemClassificationModel = {
    try {
      val fIn: FileInputStream = new FileInputStream(filename_)
      val zipIn: GZIPInputStream = new GZIPInputStream(fIn)
      val oIn: ObjectInputStream = new ObjectInputStream(zipIn)
      try return oIn.readObject.asInstanceOf[ItemClassificationModel]
      catch {
        case e: ClassNotFoundException =>
          throw new IOException("Unable to load unknown class.", e)
      } finally {
        if (fIn != null) fIn.close()
        if (zipIn != null) zipIn.close()
        if (oIn != null) oIn.close()
      }
    }
  }
}

@SerialVersionUID(0x2bb4508735311c26L)
class ItemClassificationModel(val _fitResult: FitResult[SimpleStatus, SimpleRegressor, StandardCurveType], val _settings: ItemClassifierSettings) extends ProbabilisticClassificationModel[Vector, ItemClassificationModel] {
  val paramFields: util.List[SimpleRegressor] = getParams.getUniqueRegressors
  _offsetMap = new Array[Int](paramFields.size)
  for (i <- 1 until paramFields.size) {
    val next: SimpleRegressor = paramFields.get(i)
    val index: Int = _settings.getRegressors.indexOf(next)
    if (-1 == index) throw new IllegalArgumentException("Missing regressors from fields.")
    _offsetMap(i) = index
  }
  final private var _offsetMap: Array[Int] = null
  private var _uid: String = null
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

  override val uid : String = { //Hideous hack because this is being called in the superclass constructor.
    if (null == _uid) {_uid = RandomTool.randomString(64)}
    _uid
  }

  def getFitResult: FitResult[SimpleStatus, SimpleRegressor, StandardCurveType] = _fitResult

  final def getParams: ItemParameters[SimpleStatus, SimpleRegressor, StandardCurveType] = getFitResult.getParams

  private def getModel: ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType] = {
    if (null == _model) {
      _model = new ItemModel[SimpleStatus, SimpleRegressor, StandardCurveType](getParams)
      _rawRegressors = new Array[Double](getParams.getUniqueRegressors.size)
    }
    _model
  }

  @throws[IOException]
  def save(fileName_ : String):
  Unit = {
    try {
      val fout: FileOutputStream = new FileOutputStream(fileName_)
      val zipOut: GZIPOutputStream = new GZIPOutputStream(fout)
      val oOut: ObjectOutputStream = new ObjectOutputStream(zipOut)
      try oOut.writeObject(this)
      finally {
        if (fout != null) fout.close()
        if (zipOut != null) zipOut.close()
        if (oOut != null) oOut.close()
      }
    }
  }
}