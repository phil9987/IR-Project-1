/**
  * Created by marc on 25/10/16.
  */

import breeze.linalg.{SparseVector, Vector, DenseVector}
import java.io._

class SVM(lambda: Double)
{
  val logger = new Logger("SVM")
  logger.log("Reading Data")
  val r = new ReaderTfIdfWeighted()
  val codes = scala.collection.immutable.Set[String](r.codes.toList: _*).take(1)
  var thetas = codes.map((_, DenseVector.fill[Double](r.outLength)(1.0))).toMap.par

  def updateStep(
                  theta: DenseVector[Double],
                  x: SparseVector[Double],
                  y: Double,
                  lambda: Double,
                  step: Int) = {
    val thetaShrink = theta * (1 - 1.0 / step.toDouble)
    val margin = 1.0 - (y * theta.dot(x))
    if (margin <= 0) thetaShrink
    else thetaShrink + (x * (1.0 / (lambda * step)) * y)
  }

  def storeThetaToFile(filename:String): Unit =
  {
        val pw = new PrintWriter(new File(filename))
        thetas.map { case (code, theta) => theta.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write(_))
        pw.close
  }

  def train() = {
    logger.log("training, Reading Data")
    //train SVM
    logger.log("Training SVM")
    val nrDocs = r.docCount
    var step = 1


    val docVector = r.toBagOfWords("train").toVector
    val indices = util.Random.shuffle(0 to docVector.size-1)
    for (i <- indices) {
      val dp = docVector(i)
      logger.log(s"training on documents, step = $step/$nrDocs", "trainStep", 1000)
      thetas = thetas.map { case (code, theta) => code -> updateStep(theta, dp.x,
                                                                     if (dp.y.contains(code)) 1.0
                                                                     else -1.0,lambda,step)
                          }
      step = step + 1
    }
  }

  def validate() = {
    logger.log("Validating Data")
    val validationResult = r.toBagOfWords("validation").map(dp =>
                                                              (thetas.map { case (code, theta) => (Math.signum
                                                              (theta.dot(dp.x)), code)
                                                                          }.filter(_._1 > 0).map(_._2)
                                                                .toSet, dp.y.intersect(codes) )).toList
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
                                                         }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
    logger.log("Average Precision: " + validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.length)
    logger.log("Average Recall: " + validationPrecisionRecall.map(_._2).sum / validationPrecisionRecall.length)
    logger.log("Average F1: " + validationF1.sum / validationF1.length)
  }

  def predict(filename : String) = {
    val testResult = r.toBagOfWords("test").map(dp =>  thetas.map { case (code, theta) => (Math.signum
                                                              (theta.dot(dp.x)), code)
                                                                          }.filter(_._1 > 0).map(_._2)
                                                                .toSet.mkString(dp.itemId.toString + " ", " ", "\n"))
      .toList

    val pw = new PrintWriter(new File(filename))
    testResult.foreach(pw.write)
    pw.close

  }

}


object SVM {

  def main(args: Array[String]): Unit = {
    val svm = new SVM(1e-4);
    svm.train()
    svm.storeThetaToFile("values.csv")
    svm.validate()
    svm.predict("result_svm.txt")
  }

}
