
import scala.collection.immutable.ListMap
import scala.collection.mutable.Map
import scala.collection.mutable.Set

import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  def main(args : Array[String]): Unit ={
    train("train")
    validate()
  }
  def logistic(x: Double): Double = {
    1.0 / (1.0 + Math.exp(-x))
  }


    var thetasMap : Map[String, Map[String, DenseVector[Double]]] = Map()
    var cutoffMap : Map[String, Double] = Map()

    var reader = new Reader()

    def train(setName : String): Unit = {
      thetasMap.clear()
      cutoffMap.clear()
      for (labelType <- List("topic", "industry", "country")) {
        val documents = reader.toBagOfWords(setName)

        val codes = Set[String](reader.codes.toList: _*)
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

        println("LOGREG : Starting learning...")
        var docNr = 0
        var shrink = 0
        for (doc <- documents) {
          docNr += 1
          if (docNr % 10 == 0) {
            shrink += 1
          }
          printer.print(s"LOGREG --> Updating for doc number : $docNr", 1000)
          thetasMap(labelType) = thetasMap(labelType).map {
            case (code, theta) => code -> update(theta, code, doc)
          }
          learning_rate = 1.0 / docNr
        }

        //Find optimal cutoff value such that the proportion of codes assigned is the same in the training and validation sets
        var averageCodesPerDoc = totalCodesAssigned / 50000.0
        println(s"average codes expected : $averageCodesPerDoc")
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
        println(s"cutoff value : ${cutoffMap(labelType)}")
      }
    }

    def validate() : Unit = {
      var assignedCodes: Map[Int, (Set[String], collection.immutable.Set[String])] = Map()
      //assign codes
      for (labelType <- List("topic", "industry", "country")) {
        for (validationDoc <- reader.toBagOfWords("validation")) {
          if (!assignedCodes.contains(validationDoc.itemid)) {
            assignedCodes(validationDoc.itemid) = (Set(), validationDoc.y)
          }
          assignedCodes(validationDoc.itemid) = (assignedCodes(validationDoc.itemid)._1 ++
            (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(validationDoc.x)), code)
            }.filter(_._1 > cutoffMap(labelType)).map(_._2).toSet), assignedCodes(validationDoc.itemid)._2)
        }


        val averageAssignedCodes = 1.0 * assignedCodes.map(_._2._1.size).sum / assignedCodes.size
        println(s"average codes assigned : $averageAssignedCodes") //for verification


        println("Computing score")
        val validationPrecisionRecall = assignedCodes.map { case (itemid, (actual, expected)) =>
          (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
            actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
        }
        println(s"average precision : ${validationPrecisionRecall.map(_._1).sum / validationPrecisionRecall.size}")
        println(s"average recall   : ${validationPrecisionRecall.map(_._2).sum / validationPrecisionRecall.size}")
        val validationF1 = validationPrecisionRecall
          .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
        println(s"score : ${validationF1.sum / validationF1.size}")


      }
    }

    def predict() : Unit = {



    }

}