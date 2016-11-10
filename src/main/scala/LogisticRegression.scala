
import scala.collection.immutable.ListMap

import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  def main(args: Array[String]): Unit = {
    train()
  }


    def train(): Unit = {
      val reader = new Reader(minOccurrence = 1, maxOccurrenceRate = 0.0001)
      val documents = reader.toBagOfWords("train")

      val codes = Set[String](reader.codes.toList: _*)
      var thetas = codes.map((_, DenseVector.fill[Double](reader.reducedDictionarySize + 1)(0.0))).toMap.par

      var learning_rate = 1.0

      def logistic(x: Double): Double = {
        1.0 / (1.0 + Math.exp(-x))
      }

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

      println(" LOGREG : Starting learning...")
      var docNr = 0
      for (doc <- documents) {
        docNr += 1
        printer.print(s"LOGREG --> Updating for doc number : $docNr", 1000)
        thetas = thetas.map {
          case (code, theta) => code -> update(theta, code, doc)
        }
        learning_rate = 1.0 / docNr
      }


      //Find optimal cutoff value such that the proportion of codes assigned is the same in the training and validation sets
      var averageCodesPerDoc = totalCodesAssigned / 50000.0
      var cut = new cutoffFinder(averageCodesPerDoc / codes.size) //proportion of codes assigned to total codes
      println(s"Reduced dictionary size:" + reader.reducedDictionarySize)
      val validationSet = reader.toBagOfWords("validation")
      docNr = 0
      for (validationDoc <- validationSet) {
        docNr += 1
        var codeProbability = thetas.seq.map {
          case (code, theta) => code -> logistic(theta.dot(validationDoc.x))
        }
        var sortedCodesProbabilities = ListMap(codeProbability.toSeq.sortBy(-_._2): _*)
        printer.print(s"For document number $docNr with labels ${validationDoc.y} , got : \r\n        " + sortedCodesProbabilities, 1000)
        sortedCodesProbabilities.foreach {
          case (code, prob) => cut.add(prob)
        }
      }
      var cutoff = cut.getCutoff()
      println(s"average codes expected : $averageCodesPerDoc")
      println(s"cutoff value : ${cutoff}")

      //assign codes and check that proportion of codes assigned is the same
      val validationResult = reader.toBagOfWords("validation").map(validationDoc =>
        (thetas.map { case (code, theta) => (logistic(theta.dot(validationDoc.x)), code)
        }.filter(_._1 > cutoff).map(_._2).toSet, validationDoc.y.intersect(possibleCodes.topicCodes))).toList
      val averageAssignedCodes = 1.0 * validationResult.map(_._1.size).sum / validationResult.length
      println(s"average codes assigned : $averageAssignedCodes")



      println("Computing score")
      val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
        (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
          actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
      }
      println(s"average precision : ${validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.length}")
      println(s"average recall   : ${validationPrecisionRecall.map(_._2).sum / validationPrecisionRecall.length}")
      val validationF1 = validationPrecisionRecall
        .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
      println(s"score : ${validationF1.sum / validationF1.length}")
    }

}