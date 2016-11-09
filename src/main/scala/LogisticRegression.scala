
import scala.collection.immutable.ListMap

import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  def main(args: Array[String]): Unit = {
    train()
  }


    def train(): Unit = {
      println(" LOGREG : Creating reader object..")
      val reader = new Reader()
      //val reader = new Reader(minOccurrence = 200 , maxOccurrenceRate = 0.01)
      //val documents = reader.toBagOfWords("train")
      val documents = new DataPointIterator("train", reader)

      println(" LOGREG : Creating codes and thetas ")
      val codes = Set[String](reader.codes.toList: _*)
      var thetas = collection.mutable.Map() ++ codes.map((_, DenseVector.fill[Double](reader.reducedDictionarySize + 1)(0.0))).toMap//.par

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

      /*
      println("Example theta : FASHION")
      var index : Int = 0;
      reader.dictionary.foreach {
        (t2) => printer.print(t2._1 + "  : " + thetas("GFAS")(index), 10)
          index += 1
      }
      */
      var averageCodesPerDoc = totalCodesAssigned / 50000.0
      var codesAssignedProportion = averageCodesPerDoc / codes.size
      var cut = new cutoffFinder(codesAssignedProportion) //proportion of codes assigned to total codes

      println(s"Reduced dictionary size:" + reader.reducedDictionarySize)


      val validationSet = new DataPointIterator("train", reader)
      docNr = 0
      for (validationDoc <- validationSet){
        docNr += 1
        var codeProbability = thetas.seq.map {
          case (code, theta) => code -> logistic(theta.dot(validationDoc.x))
        }
        var sortedCodesProbabilities = ListMap(codeProbability.toSeq.sortBy( - _._2):_*)
        printer.print(s"For document number $docNr with labels ${validationDoc.y} , got : \r\n        " + sortedCodesProbabilities, 1000)
        sortedCodesProbabilities.foreach {
          case (code, prob) => cut.add(prob)
        }
      }

      println(s"average codes assigned : $averageCodesPerDoc ( $codesAssignedProportion )")
      println(s"cutoff value : ${cut.getCutoff()}") }

      for (validationDoc <- validationSet){
        docNr += 1
        var codeProbability = thetas.seq.map {
          case (code, theta) => code -> logistic(theta.dot(validationDoc.x))
        }
        var sortedCodesProbabilities = ListMap(codeProbability.toSeq.sortBy( - _._2):_*)
        printer.print(s"For document number $docNr with labels ${validationDoc.y} , got : \r\n        " + sortedCodesProbabilities, 1000)
        sortedCodesProbabilities.foreach {
          case (code, prob) => cut.add(prob)
        }
      }

}