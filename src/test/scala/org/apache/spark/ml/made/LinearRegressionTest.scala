package org.apache.spark.ml.made


import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.DoubleType
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  "LRPipeline" should "precision predict" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .set_LR(0.1)
      .set_num_iter(100)

    val featureAssembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")
    val addr=getClass.getResource("/mydata.csv").getPath
    var df = spark.read.format("csv").option("header", "true").load(addr)
    df = df.withColumn("x", df("x").cast(DoubleType))
    df = df.withColumn("y", df("y").cast(DoubleType))
    df = df.withColumn("z", df("z").cast(DoubleType))
    df = df.withColumn("label", df("label").cast(DoubleType))
    val preparedData = featureAssembler.transform(df).select("features", "label")
    val labels=preparedData.select("label").collect.map(_.getDouble(0))
    val model = lr.fit(preparedData)
  }
}