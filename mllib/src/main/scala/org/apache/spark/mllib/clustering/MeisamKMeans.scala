/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import scala.util.Random

/**
 * Implementation of K-Means clustering for two dimensional data points.
 */
object MeisamKmeans {
  def MeisamKMeans(
                    points: Array[(Double, Double)],
                    k: Int,
                    maxIterations: Int)
  : (Array[(Double, Double)], Array[Int]) = {
    val rand = new Random
    val dimensions = 2
    val n = points.length
    val centers = new Array[(Double, Double)](k)

    // randomly initialize the center of clusters to random data points
    for (i <- 0 until k) {
      val randomInex = rand.nextInt(n)
      centers(i) = points(randomInex)
    }

    // Run up to maxIterations iterations of Lloyd's algorithm
    val oldClusterAssignments = Array.fill(points.length)(-1)
    var iteration = 0
    var needsUpdate = true
    while (needsUpdate && iteration < maxIterations) {
      needsUpdate = false
      val sums = Array.fill(k)((0.0, 0.0))
      val counts = Array.fill(k)(0.0)
      for ((p, i) <- points.zipWithIndex) {
        val newClusterIndex = findClosest(centers, p)
        sums(newClusterIndex) = (sums(newClusterIndex)._1 + p._1, sums(newClusterIndex)._2 + p._2)
        counts(newClusterIndex) = counts(newClusterIndex) + 1
        if (newClusterIndex != oldClusterAssignments(i)) {
          needsUpdate = true
          oldClusterAssignments(i) = newClusterIndex
        }
      }
      // Update cluster assignments
      for (i <- 0 until k) {
        // One common issue to look out for is that clusters may become empty
        // over time; a possibility to combat this is to identify such empty
        // clusters and then assign a single random point to that cluster.
        if (counts(i) == 0.0) {
          val randomIndex = rand.nextInt(n)
          val p: (Double, Double) = points(randomIndex)
          centers(i) = p
          sums(i) = (sums(i)._1 + p._1, sums(i)._2 + p._2)
          counts(i) = counts(i) + 1
          val oldClusterIndex: Int = oldClusterAssignments(randomIndex)
          sums(oldClusterIndex) = (sums(oldClusterIndex)._1 - p._1, sums(oldClusterIndex)._2 - p._2)
          counts(oldClusterIndex) = counts(oldClusterIndex) - 1
          centers(i) = p
        } else {
          centers(i) = (sums(i)._1 / counts(i), sums(i)._2 / counts(i))
        }
      }
      iteration += 1
    }
    (centers ,oldClusterAssignments)
  }

  /**
   * Returns the index to the point with closes distance from the given points.
   * @param points
   * @param point
   * @return
   */
  private def findClosest(points: Array[(Double, Double)], point: (Double, Double))
  : Int = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    for (i <- 0 until points.length) {
      val distance = euclideanDistance(point, points(i))
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
    }
    bestIndex
  }

  private def euclideanDistance(point1: (Double, Double), point2: (Double, Double)): Double = {
    val deltaX = point1._1 - point2._1
    val deltaY = point1._2 - point2._2
    math.sqrt(deltaX * deltaX + deltaY * deltaY)
  }
}

/**
 * An implementation of Expectation–maximization algorithm for 2D data points.
 */
object EMAlgorithm {

  def emAlgirithm(points: Array[(Double, Double)]
                  , k: Int // number of clusters
                  , maxIterations: Int
                  , tolerance: Double = 0.01
                   )
  : Array[(Double, Double)] = {
    val n = points.length
    val mus = new Array[(Double, Double)](k)
    val sigmas = new Array[Double](k)
    val pis = new Array[Double](k)
    val gammas = Array.ofDim[Double](n, k)
    val clusterAssignments = Array.fill[Int](n)(-1)
    initializeParams(points, mus, sigmas, pis, clusterAssignments, k)
    val maximumLikelyhoodVal = 0.0
    var converged = false
    var iterationCount = 1
    while (!converged && (iterationCount < maxIterations)) {
      iterationCount += 1
      updateGammas(points, mus, sigmas, pis, gammas)
      updateParams(points, gammas, mus, sigmas, pis, clusterAssignments)
      val newMaximumLikelyhood = maximumLiklyhood(gammas)
      converged = math.abs(maximumLikelyhoodVal - newMaximumLikelyhood) < tolerance
    }
    mus
  }

  def maximumLiklyhood(gammas: Array[Array[Double]]): Double = {
    gammas.view.flatten.sum
  }

  def initializeParams(points: Array[(Double, Double)]
                       , mus: Array[(Double, Double)]
                       , sigmas: Array[Double]
                       , pis: Array[Double]
                       , clusterAssignments: Array[Int]
                       , k: Int) = {
    val rand = new Random
    val counts = Array.fill(k)(0)
    // to make sure there is at least one data point in each cluster.
    for (i <- 0 until k) {
      clusterAssignments(i) = i
      counts(i) += 1
    }

    for (i <- k until clusterAssignments.length) {
      val randomIndex: Int = rand.nextInt(k)
      clusterAssignments(i) = randomIndex
      counts(randomIndex) += 1
    }

    for (i <- 0 until k) {
      mus(i) = points(rand.nextInt(points.length))
      sigmas(i) = 1
      pis(i) = counts(i).toDouble / counts.length
    }
  }

  def updateGammas(points: Array[(Double, Double)]
                   , mus: Array[(Double, Double)]
                   , sigmas: Array[Double]
                   , pis: Array[Double]
                   , gammas: Array[Array[Double]]
                    ): Array[Array[Double]] = {
    val k = mus.length
    val n = points.length
    for (i <- 0 until n) {

      val denominator = pis.zip(mus).zip(sigmas).map({
        case ((p, m), s) => p * normalDistribution(points(i), m, s)
      }).sum

      for (c <- 0 until k) {
        gammas(i)(c) = pis(c) * normalDistribution(points(i), mus(c), sigmas(c)) / denominator
      }
    }
    gammas
  }

  def updateParams(points: Array[(Double, Double)]
                   , gammas: Array[Array[Double]]
                   , mus: Array[(Double, Double)]
                   , sigmas: Array[Double]
                   , pis: Array[Double]
                   , clusterAssignments: Array[Int]
                    ): Unit = {
    val n = clusterAssignments.length
    val k = mus.length
    val counts = Array.fill(k)(0)
    for (i <- 0 until n) {
      counts(clusterAssignments(i)) += 1
    }
    for (c <- 0 until k) {
      var sum1 = (0.0, 0.0)
      var (sum2, sum3) = (0.0, 0.0)
      for (i <- 1 until n) {
        val norm = euclideanDistance(points(i), mus(c))
        sum1 = (sum1._1 + gammas(i)(c) * points(i)._1, sum1._2 + gammas(i)(c) * points(i)._2)
        sum2 += gammas(i)(c) * norm * norm
        sum3 += counts(c)
      }
      mus(c) = (sum1._1 / counts(c), sum1._2 / counts(c))
      sigmas(c) = sum2 / counts(c)
      pis(c) = sum3 / n
    }
  }

  def normalDistribution(point: (Double, Double), mu: (Double, Double), sigma: Double): Double = {
    val norm = euclideanDistance(point, mu)
    math.exp(-norm * norm / (2 * sigma))
  }

  private def euclideanDistance(point1: (Double, Double), point2: (Double, Double)): Double = {
    val deltaX = point1._1 - point2._1
    val deltaY = point1._2 - point2._2
    math.sqrt(deltaX * deltaX + deltaY * deltaY)
  }

}
