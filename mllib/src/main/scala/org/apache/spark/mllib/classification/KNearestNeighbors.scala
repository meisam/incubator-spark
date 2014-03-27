package org.apache.spark.mllib.classification

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Classification model trained using k-nearest neighborhood.
 *
 */
class KNearestNeighborsModel(val dataPoints: RDD[LabeledPoint], val k: Int)
  extends ClassificationModel with Serializable {

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return RDD[Int] where each entry contains the corresponding prediction
   */
  def predict(testData: RDD[Array[Double]]): RDD[Double] = {
    testData.map(predict)
  }

  def dataPointToDistance(testData: Array[Double], dataPoint: LabeledPoint) = {
    val distance: Double = MLUtils.squaredDistance(dataPoint.features, testData)
    new DistancedDataPoint(dataPoint.label, distance)
  }

  def nearestNeighbor(p1: DistancedDataPoint, p2: DistancedDataPoint) = {
    if (p1.distance < p2.distance)
      p1
    else
      p2
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return Int prediction from the trained model
   */
  def predict(testData: Array[Double]): Double = {

    val distancedRdd = dataPoints.map(dataPointToDistance(testData, _))
    val neighbor = distancedRdd.reduce(nearestNeighbor(_, _))
    neighbor.label
  }
}

class DistancedDataPoint(val label: Double, val distance: Double)
  extends Serializable {

}

object KNearestNeighbors {

  /**
   *
   * @param input RDD of (label, array of features) pairs.
   * @param k Number of neighbors to be considered in the classification
   */
  def createModel(
                   input: RDD[LabeledPoint],
                   k: Int
                   )
  : KNearestNeighborsModel = {
    new KNearestNeighborsModel(input, k)
  }
}

