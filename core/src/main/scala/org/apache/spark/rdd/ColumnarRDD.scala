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

package org.apache.spark.rdd

import scala.reflect.ClassTag

import org.apache.spark.{Logging, Partition, TaskContext}
import scala.reflect.runtime.universe._

private[spark]
class ColumnarRDD[T: ClassTag](prev: RDD[T])
  extends RDD[T](prev) {

  override def getPartitions: Array[Partition] = firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext) = {
    new ColumnarIterator[T](firstParent[T].iterator(split, context)).toIterator
  }

}

class ColumnarIterator[T](iter: Iterator[T])
  extends Iterator[T] with Logging {
  val CHUNK_SIZE: Int = 1000
  private val currentChunk: Array[Any] = new Array[Any](CHUNK_SIZE)
  private var nextToRead = CHUNK_SIZE
  private var lastIndex = -1
  private var sampleValue:Option[T] = None

  override def hasNext: Boolean = {
    iter.hasNext || (nextToRead <= lastIndex)
  }

  def encode(value: T): Any = {
    if (sampleValue == None) {
      sampleValue = Some(value)
    }

    value.hashCode()
  }

  def fetchNextChunk() = {
    logInfo("starting copying items from row-wise to columnar format...")
    for (i <- 0 to CHUNK_SIZE - 1) {
      if (iter.hasNext) {
        val currentValue = iter.next
        currentChunk(i) = encode(currentValue)
        lastIndex = i
      } else {
        // do nothing
        // TODO break out of the for loop?
      }
    }
    logInfo("... %d new elements were fetched to columnar format.".format(lastIndex))
    nextToRead = 0
  }

  override def next(): T = {
    if (nextToRead > lastIndex) {
      logInfo("going to fetch next chunk")
      fetchNextChunk()
      logInfo("fetched next chunk. The new chunk size is %d".format(currentChunk.length))
    }
    logInfo("Next to read=%d".format(nextToRead))
    nextToRead += 1
    decode(currentChunk(nextToRead - 1))
  }

  def decode(value: Any): T = {
    val className = sampleValue.getOrElse("Nonexistence").getClass.getName

    logInfo("Manifest of T is %s".format(className))
    className match {
      case "java.lang.Integer" => 1.asInstanceOf[T]
      case "java.lang.Byte" => 2.asInstanceOf[T]
      case "java.lang.Double" => 3.0.asInstanceOf[T]
      case "java.lang.String" => "String".asInstanceOf[T]
      case "scala.Tuple2$mcII$sp" => (100 , 100).asInstanceOf[T]
      case _ => throw new RuntimeException("Unknown type for %s ".format(sampleValue.toString))
    }
  }
}
