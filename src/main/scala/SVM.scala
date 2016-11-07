/**
  * Created by marc on 25/10/16.
  */

import breeze.linalg.{SparseVector, Vector, DenseVector}
import java.io._


object SVM {

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

  def main(args: Array[String]): Unit = {

    println("Reading Data")
    val r = new Reader()

    //train SVM
    println("Training SVM")
    val codes = scala.collection.immutable.Set[String](r.codes.toList: _*)
    var thetas = codes.map((_, DenseVector.fill[Double](r.reducedDictionarySize + 1)(1.0))).toMap.par
    val lambda = 1
    var step = 1
    for (dp <- r.toBagOfWords("train").take(10)) {
      thetas = thetas.map { case (code, theta) => code -> updateStep(theta, dp.x, if (dp.y.contains(code)) 1.0
      else
        -1.0,
                                                                     lambda,
                                                                     step)
                          }
      step = step + 1
    }

    //save thetas
    println("Done training. Saving found data.")
    val pw = new PrintWriter(new File("bayes.csv"))
    thetas.map { case (code, theta) => theta.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write(_))
    pw.close

    //run on validation data
    println("Running verification")
    val validationResult = r.toBagOfWords("validation").map(dp =>
                                                              (thetas.map { case (code, theta) => (Math.signum
                                                              (theta.dot(dp.x)), code)
                                                                          }.filter(_._1 > 0).map(_._2)
                                                                .toSet, dp.y)).toList

    //compute precision, recall, f1 and averaged f1
    println("Computing score")
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
                                                         }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
    println(validationF1.sum / validationF1.length)


  }

}
