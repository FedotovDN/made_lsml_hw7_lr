package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BreezeDV, sum => BreezeSum}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => mllibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasPredictionCol with PredictorParams {
  val num_iter: IntParam = new IntParam(this,"num_iter","number of iterations")
  val lr : DoubleParam = new DoubleParam(this,"learning_rate","learning rate")
  def set_num_iter(value: Int): this.type=set(num_iter,value)
  setDefault(num_iter,100)
  def set_LR(value: Double):this.type=set(lr, value)
  setDefault(lr,0.1)
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getPredictionCol))
    }

    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getPredictionCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("LR"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder4Vector: Encoder[Vector] = ExpressionEncoder()
    implicit val encoder4Double: Encoder[Double] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Double)] = dataset.select(dataset($(featuresCol)).as[Vector], dataset($(labelCol)).as[Double])
    val nFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: BreezeDV[Double] = BreezeDV.zeros[Double](nFeatures)

    for (i <- 0 to $(num_iter)) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v._1.asBreeze
          val y = v._2
          val grad = X * (BreezeSum(X * weights) - y)
          summarizer.add(mllibVectors.dense(grad.toArray))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      weights=weights-$(lr)*summary.mean.asBreeze
    }
      copyValues(new LinearRegressionModel(uid,Vectors.fromBreeze(weights).toDense).setParent(this))

    }
    override def copy(extra: ParamMap): Estimator[LinearRegressionModel]
    = defaultCopy(extra)
    override def transformSchema(schema: StructType): StructType
    = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           weights: Vector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable with PredictorParams {


  private[made] def this(weights: DenseVector) =
    this(Identifiable.randomUID("LRModel"), weights)

  def predict(features: DenseVector): Double = {

    features.dot(weights)
  }

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(uid, weights))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        println(x)
        Vectors.fromBreeze((BreezeDV(weights.asBreeze.dot(x.asBreeze)))) // /:/ bStds)
      })
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(Tuple1(weights.asInstanceOf[Vector]))).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")
      implicit val encoder4Vector: Encoder[Vector] = ExpressionEncoder()
      implicit val encoder4Double: Encoder[Double] = ExpressionEncoder()

      val weights = vectors.select(vectors("_1").as[Vector]).first()
      val model = new LinearRegressionModel(weights.toDense)
      metadata.getAndSetParams(model)
      model
    }
  }
}
