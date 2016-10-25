/**
  * Created by marc on 25/10/16.
  */

import scala.util.Random
import breeze.linalg.{Vector, DenseVector}
case class DataPoint(x: Vector[Double], y: Double)


object SVM {

  def updateStep (
                   theta: DenseVector[Double],
                   p: DataPoint,
                   lambda : Double,
                   step: Int) = {
    val thetaShrink = theta * (1-1.0/step.toDouble)
    val margin = 1.0 - ( p.y * theta.dot(p.x) )
    if (margin <= 0) thetaShrink
    else thetaShrink + (p.x *( 1.0 / (lambda * step)) * p.y)
  }

  def main(args: Array[String]): Unit = {
    //testData = Seq.fill(n)(Random.nextInt)
    val N = 2000
    val trainingData =
        (1 to N)
          .map(x => (Random.nextInt(100),Random.nextInt(100),Random.nextInt(100)))
        .map(x => DataPoint(Vector[Double](x._1, x._2, x._3, 1), if (x._1+x._2+x._3 < 100) 1.0 else -1.0 ) )
      .toStream

    val testData =
      (1 to N)
        .map(x => (Random.nextInt(100),Random.nextInt(100),Random.nextInt(100)))
        .map(x => DataPoint(Vector[Double](x._1, x._2, x._3, 1), if (x._1+x._2+x._3 < 100) 1.0 else -1.0 ) )
        .toStream

    var theta = DenseVector[Double](1.0, 1.0, 1.0, 1.0)
    val lambda = 100
    var step = 1

    print("Training SVM\n")
    trainingData.foreach { p => theta = updateStep(theta, p, lambda, step)
                                step  = step + 1 }

    print(s"Theta = $theta\n")

    print("Testing SVM\n")
    val result = testData.map(p => (Math.signum(theta.dot(p.x)), p.y)).toList
    val resultCount = result.map( x => x._1 * x._2 ).groupBy(x => x).mapValues(_.size)
    print(result)
    val p = resultCount(1.0)*1.0/(resultCount(-1.0) + resultCount(1.0))
    print(s"\n$p\n")

  }

}




