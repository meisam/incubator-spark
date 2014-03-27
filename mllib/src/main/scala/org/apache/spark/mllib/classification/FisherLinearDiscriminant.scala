package org.apache.spark.mllib.classification

import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.jblas.DoubleMatrix

/**
 * Created by Fathi on 3/27/14.
 */
class FisherLinearDiscriminantModel(
                                     override val weights: Array[Double],
                                     override val intercept: Double)
  extends GeneralizedLinearModel(weights, intercept)
  with ClassificationModel with Serializable {


  override def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
                            intercept: Double) = {
    dataMatrix.mmul(weightMatrix).get(0) + intercept
  }
}

/**
  */
class FisherLinearDiscriminant
  extends Serializable {

  def createModel(weights: Array[Double], intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
  */
object FisherLinearDiscriminant {

  def matrixAdd(matrix1: DoubleMatrix, matrix2: DoubleMatrix): DoubleMatrix =
    matrix1.add(matrix2)

  /**
    */
  def train(
             input: RDD[LabeledPoint],
             f: LabeledPoint => Boolean)
  : FisherLinearDiscriminantModel = {

    val belongingDataPoints = input.filter(f).map(dp => new DoubleMatrix(dp.features))
    val notBelongingDataPoints = input.filter(!f(_)).map(dp => new DoubleMatrix(dp.features))
    val m1 = belongingDataPoints.reduce((x, y) => x.add(y))
    val m2 = notBelongingDataPoints.reduce((x, y) => x.add(y))

    val s1: DoubleMatrix = belongingDataPoints.map(x => {
      val z = x.sub(m1);
      z.mmul(z.transpose())
    }).reduce(matrixAdd)

    val s2 = notBelongingDataPoints.map(x => {
      val z = x.sub(m2);
      z.mmul(z.transpose())
    }).reduce(matrixAdd)

    val Sw = s1.add(s2)
    val w = Sw.mmul(m2.sub(m1))
    val intercept: Double = -(w.transpose().mmul(m1.add(m2))).get(0) / 2

    new FisherLinearDiscriminantModel(w.toArray, intercept)
  }


  def main(args: Array[String]) {
    if (args.length != 4) {
      println("Usage: LogisticRegression <master> <input_dir> <step_size> " +
        "<niters>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "LogisticRegression")
    val data = MLUtils.loadLabeledData(sc, args(1))
    val model = LogisticRegressionWithSGD.train(data, args(3).toInt, args(2).toDouble)
    println("Weights: " + model.weights.mkString("[", ", ", "]"))
    println("Intercept: " + model.intercept)

    sc.stop()
  }
}
