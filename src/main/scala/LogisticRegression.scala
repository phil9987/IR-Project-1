
import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

    val ITERATIONS = 1

  def main(args: Array[String]): Unit = {
    train()
  }

    def train(): Unit = {
      println("Creating reader object..")

      val reader = new Reader()

      println("Creating document iterator for training set..")
      val documents = reader.toBagOfWords("train")

      val codes = scala.collection.immutable.Set[String](reader.codes.toList: _*)
      var thetas = codes.map((_, DenseVector.fill[Double](reader.reducedDictionarySize + 1)(0.0))).toMap // .par

      var learning_rate = 1

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

      println("Starting learning...")
      var docNr = 0
      for (iteration <- 1 to ITERATIONS) {
        docNr = 0
        for (doc <- documents) {
          docNr += 1
          println(s"---- Updating for doc number : $docNr")
          thetas.map {
            case (code, theta) => code -> update(theta, code, doc)
          }
        }

      }
    }
}