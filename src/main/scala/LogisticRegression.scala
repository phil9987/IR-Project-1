
import scala.collection.immutable.ListMap
import scala.collection.mutable.Map
import scala.collection.mutable.Set

import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  val logger = new Logger("LogisticRegression")

  def main(args : Array[String]): Unit ={
    train("train")
    validate()
  }
  def logistic(x: Double): Double = {
    1.0 / (1.0 + Math.exp(-x))
  }

    var labelTypes = List("topic", "industry", "country")

    var thetasMap : Map[String, Map[String, DenseVector[Double]]] = Map()
    var cutoffMap : Map[String, Double] = Map()

    var reader = new Reader(minOccurrence = 1, maxOccurrenceRate = 0.2)

    def train(setName : String): Unit = {
      thetasMap.clear()
      cutoffMap.clear()
      for (labelType <- labelTypes) {

        val documents = reader.toBagOfWords(setName)

        val codes = Set[String](reader.codes.toList: _*).intersect(possibleCodes.fromString(labelType))
        thetasMap(labelType) = Map() ++ codes.map((_, DenseVector.fill[Double](reader.reducedDictionarySize + 1)(0.0))).toMap
        var learning_rate = 1.0

        var totalCodesAssigned = 0
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
        var shrink = 0
        for (doc <- documents) {
          docNr += 1
          if (docNr % 10 == 0) {
            shrink += 1
          }
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
        val validationSet = reader.toBagOfWords("validation")
        docNr = 0
        for (validationDoc <- validationSet) {
          docNr += 1
          var codeProbability = thetasMap(labelType).seq.foreach {
            case (code, theta) => cut.add(logistic(theta.dot(validationDoc.x)))
          }
        }
        cutoffMap(labelType) = cut.getCutoff()
        logger.log(s"cutoff value : ${cutoffMap(labelType)}")
      }
    }

    def validate() : Unit = {
      var assignedCodes: Map[Int,Set[String]] = Map()
      var realCodes : Map[Int, collection.immutable.Set[String]] = Map()
      //assign codes
      for (labelType <- labelTypes) {
        for (validationDoc <- reader.toBagOfWords("validation")) {
          if (!assignedCodes.contains(validationDoc.itemId)) {
            assignedCodes(validationDoc.itemId) = Set()
            realCodes(validationDoc.itemId) = validationDoc.y
          }
          assignedCodes(validationDoc.itemId) ++= //adds the codes
            (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(validationDoc.x)), code)
            }.filter(_._1 > cutoffMap(labelType)).map(_._2).toSet)
        }


        val averageAssignedCodes = 1.0 * assignedCodes.map(_._2.size).sum / assignedCodes.size
        logger.log(s"average codes assigned per doc in total: $averageAssignedCodes") //for verification

        logger.log("Computing score")
        val validationPrecisionRecall = assignedCodes.map { case (itemId, assigned) =>
          (assigned.intersect(realCodes(itemId)).size.toDouble / (assigned.size + scala.Double.MinPositiveValue),
            assigned.intersect(realCodes(itemId)).size.toDouble / (realCodes(itemId).size + scala.Double.MinPositiveValue))

        }
        logger.log(s"average precision : ${validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.size}")
        logger.log(s"average recall   : ${validationPrecisionRecall.map(_._2).sum / validationPrecisionRecall.size}")
        val validationF1 = validationPrecisionRecall
          .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
        logger.log(s"score : ${validationF1.sum / validationF1.size}")


      }
    }

    def predict() : Unit = {



    }

}