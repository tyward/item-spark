package org.apache.spark.ml.ann

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD


object AnnUtil {


  def trainAndExtractWeights(trainer: FeedForwardTrainer, data : RDD[(Vector, Vector)]): Vector = {
    val (mlpModel, someVector)  = trainer.train(data)
    return mlpModel.weights
  }



}
