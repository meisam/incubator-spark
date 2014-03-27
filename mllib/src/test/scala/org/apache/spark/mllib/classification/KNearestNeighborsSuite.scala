package org.apache.spark.mllib.classification

import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite
import org.apache.spark.mllib.util.{MLUtils, LocalSparkContext}


object KNearestNeighborsSuite {

  private def calcLabel(p: Double, pi: Array[Double]): Int = {
    var sum = 0.0
    for (j <- 0 until pi.length) {
      sum += pi(j)
      if (p < sum) return j
    }
    -1
  }

  def generateInput(nPoints: Int, x1Bias: Double, x2Bias: Double): Seq[LabeledPoint] = {
    var seq: Seq[LabeledPoint] = Seq.empty
    for (i <- 0 until nPoints) {
      for (j <- 0 until nPoints) {
        val x1: Double = i.toDouble + x1Bias
        val x2: Double = j.toDouble + x2Bias
        val label = if ((x1 * x2) >= 0) 1 else -1
        seq = seq ++ Seq(new LabeledPoint(label, Array(x1, x2)))
      }
    }
    seq
  }
}

class KNearestNeighborsSuite extends FunSuite with LocalSparkContext {

  def validatePrediction(predictions: Seq[Double], input: Seq[LabeledPoint]) {
    val numOfPredictions = predictions.zip(input).count {
      case (prediction, expected) =>
        prediction != expected.label
    }
    // At least 80% of the predictions should be on.
    assert(numOfPredictions < input.length / 5)
  }

  test("1-nearest neighbor") {

    val testData: Seq[Array[Double]] = KNearestNeighborsSuite.generateInput(3, -1.4, -1.3).map(_.features)
    val testRDD = sc.parallelize(testData, 2)
    testRDD.cache()

    val validationData = KNearestNeighborsSuite.generateInput(6, -3.0, -3.0)
    val validationRDD = sc.parallelize(validationData, 2)
    val model: KNearestNeighborsModel = KNearestNeighbors.createModel(validationRDD, 1)

    println("validation data")
    for (x <- validationRDD.collect()) {
      println("Label: %.3f, values: %s".format(x.label, x.features.mkString(",")))
    }

    println("test data")
    for (x <- testRDD.collect()) {
      println("Label: ----, values: %s".format(x.mkString(",")))
    }

    // Test prediction on RDD.

    println("Predictions")
    for (x <- testData) {
      val prediction = model.predict(x)
      println("Label: %.3f, values: %s".format(prediction, x.mkString(",")))
    }
  }

  test("distance") {
    val distance = MLUtils.squaredDistance(Array(0d, 0d), Array(3d, 4d))
    println("Distance is %.3f".format(distance))
  }

  val epsilon = 0.01

  test("classify wine data") {
    val baseDir = "/home/fathi/Courses/CSE5522-Kulis/homework/hw4/"

    val trainFeaturesRdd = sc.textFile(baseDir + "wine_train_x.txt", 1).map(x => x.split("\\s+")).map(x => x.map(_.toDouble))
    //    println(trainFeaturesRdd.map(_.mkString("[", ",", "]")).collect().mkString("\n"))

    val trainLabelsRdd = sc.textFile(baseDir + "wine_train_y.txt", 1).map(_.toDouble)
    //        println(trainLabelsRdd.collect().mkString("\n"))

    val trainDataPointsDataset = trainLabelsRdd.zip(trainFeaturesRdd).map(x => new LabeledPoint(x._1, x._2))
    //    println(trainDataPointsDataset.collect().mkString("\n"))

    val model = KNearestNeighbors.createModel(trainDataPointsDataset, 1)

    val testFeatureDataset = sc.textFile(baseDir + "wine_test_x.txt", 1).map(x => x.split("\\s+")).map(x => x.map(_.toDouble))
    //    println(testFeatureDataset.map(_.mkString("[", ",", "]")).collect().mkString("\n"))

    val clusteringResults = testFeatureDataset.collect().map(model.predict(_))
    println(clusteringResults.mkString("Clusterring results:---------------\n", ", ", "\n---------end of clustering results."))


    val testLabelsDataset = sc.textFile(baseDir + "wine_test_y.txt", 1).map(_.toDouble)
    println(testLabelsDataset.collect().mkString(", "))

    val expectedResults = testLabelsDataset.collect()

    val resultCompare = clusteringResults.zip(expectedResults).map(x => Math.abs(x._1 - x._2) < epsilon).map(if (_) 1 else 0).reduce(_ + _)

    println("Percent of correct results: %f".format(100.00 * resultCompare / clusteringResults.size))
    //    val clusteringResults = model.predict(testFeatureDataset)
    //    println(clusteringResults.collect().mkString("\n"))

  }

  test("fisher linear discriminant") {
    val baseDir = "/home/fathi/Courses/CSE5522-Kulis/homework/hw4/"

    val trainFeaturesRdd = sc.textFile(baseDir + "wine_train_x.txt", 1).map(x => x.split("\\s+")).map(x => x.map(_.toDouble))
    //    println(trainFeaturesRdd.map(_.mkString("[", ",", "]")).collect().mkString("\n"))

    val trainLabelsRdd = sc.textFile(baseDir + "wine_train_y.txt", 1).map(_.toDouble)
    //        println(trainLabelsRdd.collect().mkString("\n"))

    val trainDataPointsDataset = trainLabelsRdd.zip(trainFeaturesRdd).map(x => new LabeledPoint(x._1, x._2))
    //    println(trainDataPointsDataset.collect().mkString("\n"))

    val model = FisherLinearDiscriminant.train(trainDataPointsDataset, x => x.label < 1.1)

    val testFeatureDataset = sc.textFile(baseDir + "wine_test_x.txt", 1).map(x => x.split("\\s+")).map(x => x.map(_.toDouble))
    //    println(testFeatureDataset.map(_.mkString("[", ",", "]")).collect().mkString("\n"))

    val clusteringResults = testFeatureDataset.collect().map(model.predict(_))
    println(clusteringResults.mkString("Clusterring results:---------------\n", ", ", "\n---------end of clustering results."))


    val testLabelsDataset = sc.textFile(baseDir + "wine_test_y.txt", 1).map(_.toDouble)
    println(testLabelsDataset.collect().mkString(", "))

    val expectedResults = testLabelsDataset.collect()

    val resultCompare = clusteringResults.zip(expectedResults).map(x => Math.abs(x._1 - x._2) < epsilon).map(if (_) 1 else 0).reduce(_ + _)

    println("Percent of correct results: %f".format(100.00 * resultCompare / clusteringResults.size))
    //    val clusteringResults = model.predict(testFeatureDataset)
    //    println(clusteringResults.collect().mkString("\n"))

  }

}
