/**
  * Created by marc on 25/10/16.
  */

import breeze.linalg.{SparseVector, Vector, DenseVector}
import java.io._

class SVM(lambda: Double)
{
  val logger = new Logger("SVM")
  logger.log("Reading Data")
  val r = new TfIDfReader(0)
  val codes = scala.collection.immutable.Set[String](r.codes.toList: _*)
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

  def train() = {
    logger.log("training, Reading Data")
    //train SVM
    logger.log("Training SVM")
    val nrDocs = r.docCount
    var step = 1
    for (dp <- r.toBagOfWords("train")) {
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
                                                                .toSet, dp.y)).toList
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
                                                         }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
    logger.log("Average Precision: " + validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.length)
    logger.log("Average Recall: " + validationPrecisionRecall.map(_.__2).sum / validationPrecisionRecall.length)
    logger.log("Average F1: " + validationF1.sum / validationF1.length)
  }

  def predict() = {}

}


object SVM {


  def main(args: Array[String]): Unit = {

//    //save thetas
//    logger.log("Done training. Saving found data.")
//    val pw = new PrintWriter(new File("bayes.csv"))
//    thetas.map { case (code, theta) => theta.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write(_))
//    pw.close
//
//    //run on validation data
//    logger.log("Running verification")
//
//
//    //compute precision, recall, f1 and averaged f1
//    logger.log("Computing score")

    val svm = new SVM(0.01);
    svm.train()
    svm.validate()

  }

}
