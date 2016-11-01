/**
  * Created by marc on 25/10/16.
  */

import breeze.linalg.{SparseVector, Vector, DenseVector}

//case class DataPoint(x: Vector[Double], y: Vector[Double])


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

    val r = new Reader()
    val codes = r.codes


    val currentCode = codes.head
    var theta = DenseVector.fill[Double](r.reducedDictionarySize + 1)(1.0)
    val lambda = 1
    var step = 1
    for (dp <- r.toBagOfWords("train")) {
      theta = updateStep(theta, dp.x, if (dp.y.contains(currentCode)) 1.0 else -1.0, lambda, step)
      step = step + 1
    }

    val validationResult = r.toBagOfWords("validation").map(dp => (Math.signum(theta.dot(dp.x)),
      if (dp.y.contains(currentCode)) 1.0 else -1.0 ))
    val validationResultCount = validationResult.map(x => x._1 * x._2).groupBy(x => x).mapValues(_.size)
    println(validationResultCount)
    val p = validationResultCount(1.0) * 1.0 / (validationResultCount(-1.0) + validationResultCount(1.0))
    println(p)

  }

}
