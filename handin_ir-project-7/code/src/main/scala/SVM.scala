/**
  * Created by marc on 25/10/16.
  */

import breeze.linalg.{SparseVector, DenseVector}
import java.io._

/**
  * Creates an SVM, with the given parameter.
  * @param lambda Lambda regulation parameter for the SVM algorithm.
  */
class SVM(lambda: Double)
{
  val logger = new Logger("SVM")
  logger.log("Reading Data")

  //use this epislon to avoid division by 0
  val eps = 1e-5

  //the reader used for the SVM
  val r = new ReaderTfIdfWeighted(30,1,false)
  //already incorporates the pre-processing
  //different choice of reader and its arguments results in different pre-processing
  val codes = scala.collection.immutable.Set[String](r.codes.toList: _*)

  //Dictionary that saves a parameter vector (and thereby state of the SVM) for each code
  var thetas = codes.map((_, DenseVector.fill[Double](r.outLength)(1.0))).toMap.par

  /**
    * Performs a single updated step from for a SVM, given the current state and a
    * data point.
    * Mostly taken from lecture code.
    * @param theta  Parameter vector of the SVM.
    * @param x      Feature vector of the data point.
    * @param y      1.0/-1.0 label of the data point.
    * @param lambda Lambda parameter
    * @param step   Current time step.
    * @return       The updated parameter vector theta.
    */
  private def updateStep(
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

  /**
    * Stores the current state (thetas) to a file, for later reuse.
    * @param filename Name of the file.
    */
  def storeThetaToFile(filename:String): Unit =
  {
        val pw = new PrintWriter(new File(filename))
        thetas.map { case (code, theta) => theta.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write)
        pw.close()
  }

  /**
    * Traines the SVM.
    */
  def train() = {
    logger.log("training, Reading Data")
    logger.log("Training SVM")
    val nrDocs = r.docCount
    var step = 1

    //In order to enable random order training replace the head of the for loop
    //with this
    //    val docVector = r.toBagOfWords("train").toVector
    //    val indices = util.Random.shuffle(0 to docVector.size-1)
    //    for (i <- indices) {
    //      val dp = docVector(i)

    //for each document in the training data...
    for(dp <- r.toBagOfWords("train")) {
      logger.log(s"training on documents, step = $step/$nrDocs", "trainStep", 1000)
      val x = breeze.linalg.normalize(dp.x)
      //...update the SVMs of all classes, this map is quite efficient as thetas is a
      //parallel map - thus allowing good cpu utilization for training
      thetas = thetas.map { case (code, theta) => code -> updateStep(theta, x,
                                                                     if (dp.y.contains(code)) 1.0
                                                                     else -1.0,lambda,step)
                          }
      //advance time stemp
      step = step + 1
    }
  }

  /**
    * Evaluate the SVM for the data points in the validation data
    * and compute precession, recall and F1 score.
    */
  def validate() = {
    logger.log("Validating Data")

    //for each word in validation set, predict labels
    val validationResult = r.toBagOfWords("validation").map(dp =>
                                                              {(thetas.map {
                                                                            case (code, theta) =>
                                                                              (code, Math.signum(theta.dot(
                                                                                breeze.linalg.normalize(dp.x))))
                                                                          }.filter(_._2 > 0).keys
                                                                .toSet, dp.y.intersect(codes) )}).toList

    //calculate precession and recall for each document
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
       (actual.intersect(expected).size.toDouble / (actual.size + eps),
        actual.intersect(expected).size.toDouble / (expected.size + eps))
                                                         }
    //calculate F1 score for each document
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + eps) }

    logger.log("Average Precision: " + validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.length)
    logger.log("Average Recall: " + validationPrecisionRecall.map(_._2).sum / validationPrecisionRecall.length)
    logger.log("Average F1: " + validationF1.sum / validationF1.length)
    List[Double](validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.length,
                 validationPrecisionRecall.map(_
                                                                                                                      ._2)
      .sum / validationPrecisionRecall.length,
      validationF1.sum / validationF1.length)
  }

  /**
    * Predict class for the test file set and write them to an output file.
    * @param filename Name of the output file.
    */
  def predict(filename : String) = {

    //for each word in
    val testResult = r.toBagOfWords("test").map(dp =>  thetas.map { case (code, theta) => (code, Math.signum
                                                              (theta.dot(breeze.linalg.normalize(dp.x))))
                                                                          }.filter(_._2 > 0).keys
                                                                .toSet.mkString(dp.itemId.toString + " ", " ", "\n"))
      .toList

    val pw = new PrintWriter(new File(filename))
    testResult.foreach(pw.write)
    pw.close()

  }

}

/**
  * Companion Object with main method.
  * Used for internal testing of SVM.
  */
object SVM {
  def main(args: Array[String]): Unit = {

    val svm = new SVM(1e-5)
    svm.train()
    //svm.storeThetaToFile("ir-2016-1-project-7-svm.csv")
    svm.validate()
    svm.predict("ir-2016-1-project-7-svm.txt")

  }
}
