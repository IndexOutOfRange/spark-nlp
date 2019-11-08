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

package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}

class SparkNLPRandomForestClassifier(override val uid: String)
  extends RandomForestClassifier(uid: String) {

  def this() = this(Identifiable.randomUID("rfc"))

  override def fit(dataset: Dataset[_]): SparkNLPRandomForestClassificationModel = {
    super.fit(dataset).asInstanceOf[SparkNLPRandomForestClassificationModel]
  }
}

object SparkNLPRandomForestClassifier extends RandomForestClassifier

class SparkNLPRandomForestClassificationModel (
                                                @Since("1.5.0") override val uid: String,
                                                private val _trees: Array[DecisionTreeClassificationModel],
                                                @Since("1.6.0") override val numFeatures: Int,
                                                @Since("1.5.0") override val numClasses: Int)
  extends RandomForestClassificationModel(uid, _trees, numFeatures, numClasses)
    with RFPublicPredictRaw {
  require(_trees.nonEmpty, "SparkNLPRandomForestClassificationModel requires at least 1 tree.")

}


trait RFPublicPredictRaw {

  def predictRaw(vector: Vector): Vector

  def raw2prediction(vector: Vector): Double

  def raw2probability(vector: Vector): Vector

  def predictRawPublic(features: Vector): (Double, Map[String, String]) = {
    val raw = predictRaw(features)
    (raw2prediction(raw), Map("probabilities" -> raw2probability(raw).toString))
  }
}