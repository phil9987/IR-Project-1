
import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  def main(args: Array[String]): Unit = {
    train()
  }


    def train(): Unit = {
      println(" LOGREG : Creating reader object..")

      val reader = new Reader(minOccurrence = 200, maxOccurrenceRate = 0.01)
      //val documents = reader.toBagOfWords("train")
      val documents = new DataPointIterator("train", reader)

      println(" LOGREG : Creating codes and thetas ")
      val codes = Set[String](reader.codes.toList: _*)
      var thetas = codes.map((_, DenseVector.fill[Double](reader.reducedDictionarySize + 1)(0.0))).toMap.par

      var learning_rate = 1.0

      def logistic(x: Double): Double = {
        1.0 / (1.0 + Math.exp(-x))
      }

      def update(theta: DenseVector[Double], code: String, doc: DataPoint): DenseVector[Double] = {
        var alpha = reader.getProbabilityOfCode(code)
        if (doc.y contains code) {
          theta - doc.x * (learning_rate * (1 - alpha) * (1 - logistic(theta.dot(doc.x))))
        }
        else {
          theta + doc.x * (learning_rate * alpha * (logistic(theta.dot(doc.x))))
        }
      }

      println(" LOGREG : Starting learning...")
      var docNr = 0
      for (doc <- documents) {
        docNr += 1
        println(s"LOGREG --> Updating for doc number : $docNr")
        thetas = thetas.map {
          case (code, theta) => code -> update(theta, code, doc)
        }
        learning_rate = 1.0 / docNr
      }

      println(thetas)
      println("Example theta : FASHION")
      var index : Int = 0;
      reader.dictionary.foreach {
        (t2) => println(t2._1 + "  : " + thetas("GFAS")(index))
          index += 1
      }


    }
}