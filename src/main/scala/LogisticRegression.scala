
import scala.collection.immutable.ListMap
import scala.collection.mutable.Map
import scala.collection.mutable.Set


import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  val logger = new Logger("LogisticRegression")

  def main(args : Array[String]): Unit ={
    train("train")

    //adjust the country cutoff (performance better than the computed cutoff)
    validate("validation")
    predict("test")
  }

  def logistic(x: Double): Double = {
    1.0 / (1.0 + Math.exp(-x))
  }

  //the types of labels to train
  var labelTypes = List("topic","country")
  //for a given labelType, holds the map from codes (string) to theta vectors
  var thetasMap : Map[String, Map[String, DenseVector[Double]]] = Map()
  //for a given labelType, holds the map from codes (string) to the probability cutoff
  var cutoffMap : Map[String, Double] = Map()
  //the reader to use
  var reader = new Reader(300, 1, true)

  /**
  * Trains the model on the dataset specified.
  * This means that it will change the theta values in the thetasMap as well as the cutoff in cutoffMap
  * @param setName : The set name to use, i.e. train most of the time
  */
  def train(setName : String = "train"): Unit = {
    thetasMap.clear()
    cutoffMap.clear()
    for (labelType <- labelTypes) {
      val documents = reader.toBagOfWords(setName)
      val codes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString(labelType))
      thetasMap(labelType) = Map() ++ codes.map((_, DenseVector.fill[Double](reader.outLength)(0.0))).toMap

      var learning_rate = 1.0
      var totalCodesAssigned = 0

      /**
      * Returns updated theta vector when considering a new document
      *  @param theta : The existing theta value
      *  @param code : The code of the theta value
      *  @param doc : The new document input
      *  @return : The new theta value
      * */
      def update(theta: DenseVector[Double], code: String, doc: DataPoint): DenseVector[Double] = {
        val alpha = reader.getProbabilityOfCode(code)
        if (doc.y contains code) {
          totalCodesAssigned += 1
          return theta + doc.x * (learning_rate * (1 - alpha) * (1 - logistic(theta.dot(doc.x))))
        }
        else {
          return theta - doc.x * (learning_rate * alpha * (logistic(theta.dot(doc.x))))
        }
      }

      logger.log(s"Starting learning theta values for label $labelType...")
      var docNr = 0
      for (doc <- documents) {
        docNr += 1
        logger.log(s"Updating for doc number : $docNr", "updateFor", 1000)
        thetasMap(labelType) = thetasMap(labelType).map {
          case (code, theta) => code -> update(theta, code, doc)
        }
        learning_rate = 1.0 / docNr
      }

      //Find optimal cutoff value such that the proportion of codes assigned is the same in the training and validation sets
      var averageCodesPerDoc = totalCodesAssigned / 50000.0
      logger.log(s"average codes expected for type $labelType: $averageCodesPerDoc")
      var cut = new cutoffFinder(averageCodesPerDoc / codes.size) //proportion of codes assigned to total codes
      for (validationDoc <- reader.toBagOfWords("validation")) {
        var codeProbability = thetasMap(labelType).seq.foreach {
          case (code, theta) => cut.add(logistic(theta.dot(validationDoc.x)))
        }
      }
      cutoffMap(labelType) = cut.getCutoff()
      logger.log(s"optimal cutoff value : ${cutoffMap(labelType)}")
    }
  }

  /**
    * Validates the trained model on the dataset specified. The dataset must have the real codes
    * so that performance can be evaluated.
    *
    * @return : The F1-score obtained
    */
  def validate(setName : String) : Double = {
    var assignedCodes: Map[Int,Set[String]] = Map()
    var realCodes : Map[Int, Set[String]] = Map()
    //assign codes
    for (labelType <- labelTypes) {
      for (validationDoc <- reader.toBagOfWords(setName)) {
        if (!assignedCodes.contains(validationDoc.itemId)) {
          assignedCodes(validationDoc.itemId) = Set()
          realCodes(validationDoc.itemId) = Set(Codes.fromString(labelType).toArray:_*).intersect(validationDoc.y)
        }
        assignedCodes(validationDoc.itemId) ++= //adds the codes
          (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(validationDoc.x)), code)
          }.filter(_._1 > cutoffMap(labelType)).map(_._2).toSet)
      }
    }

    //compare predicted values with read values
    var buf = scala.collection.mutable.ListBuffer.empty[Tuple2[Set[String], Set[String]]]
    assignedCodes.foreach {
      case (itemid, assigned) =>
      buf += new Tuple2(assignedCodes(itemid) , realCodes(itemid))
    }

    logger.log(s"average codes assigned per doc in total: ${1.0 * assignedCodes.map(_._2.size).sum / assignedCodes.size}")


    logger.log("Computing score")
    val validationPrecisionRecall = buf.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
    }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue)}
    logger.log(s"average precision : ${validationPrecisionRecall.map(_._1).sum / 10000.0}")
    logger.log(s"average recall   :  ${validationPrecisionRecall.map(_._2).sum / 10000.0}")
    logger.log(s"score : ${validationF1.sum / 10000.0}")
    return (validationF1.sum / 10000.0)
  }

  /**
    * Fits the data to  the trained model. Creates a file with the predictions.
    */
  def predict(setName : String) : Unit = {
      var assignedCodes: Map[Int,Set[String]] = Map()
      //assign codes
      for (labelType <- labelTypes) {
        for (testDoc <- reader.toBagOfWords(setName)) {
          if (!assignedCodes.contains(testDoc.itemId)) {
            assignedCodes(testDoc.itemId) = Set()
          }
          assignedCodes(testDoc.itemId) ++= //adds the codes
            (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(testDoc.x)), code)
            }.filter(_._1 > cutoffMap(labelType)).map(_._2).toSet)
        }

      }

      import java.io.PrintWriter
      import java.io.File
      val pw = new PrintWriter(new File("prediction.txt"))
      assignedCodes.toSeq.sortBy(_._1).foreach{ case(id, codes) =>
        var line = s"$id "
        codes.foreach(x=> line = line.concat(x + " "))
        pw.write(line + "\r\n")
      }
      pw.close()
    }
}