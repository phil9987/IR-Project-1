
import scala.collection.immutable.ListMap
import scala.collection.mutable.Map
import scala.collection.mutable.Set

import breeze.linalg.{SparseVector, Vector, DenseVector}

object LogisticRegression{

  val logger = new Logger("LogisticRegression")

  def main(args : Array[String]): Unit ={

    train("train")
    /*
    var  possiblecutoffs = List(0.50, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.60)
    var resultingScores = scala.collection.mutable.Buffer(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    var index = 0
    for (cutoff  <- possiblecutoffs) {
      resultingScores(index) = validate(cutoff)
      index += 1
    }

    validate("0.525")
    println(s"RESULTING SCORES : $resultingScores")
    */
    validate(cutoff = 0.525)
    predict(cutoff = 0.525)
  }
  def logistic(x: Double): Double = {
    1.0 / (1.0 + Math.exp(-x))
  }

 // var labelTypes = List("topic", "industry", "country")
  var labelTypes = List("country")

  var thetasMap : Map[String, Map[String, DenseVector[Double]]] = Map()
  var cutoffMap : Map[String, Double] = Map()

  var reader = new TitleReader(20 , 1, true)
  //var reader = new TfIDfReader(40000)

  def train(setName : String): Unit = {
    thetasMap.clear()
    cutoffMap.clear()
    for (labelType <- labelTypes) {

      val documents = reader.toBagOfWords(setName)
      val codes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString(labelType))
      thetasMap(labelType) = Map() ++ codes.map((_, DenseVector.fill[Double](reader.outLength)(0.0))).toMap
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
      logger.log(s"cutoff value : ${cutoffMap(labelType)}")
    }
  }

  def validate(cutoff: Double = -1) : Double = {
    var myCut = cutoff
    var assignedCodes: Map[Int,Set[String]] = Map()
    var realCodes : Map[Int, Set[String]] = Map()
    //assign codes
    for (labelType <- labelTypes) {
      if (myCut == -1){
        myCut =cutoffMap(labelType)
      }

      for (validationDoc <- reader.toBagOfWords("validation")) {
        if (!assignedCodes.contains(validationDoc.itemId)) {
          assignedCodes(validationDoc.itemId) = Set()
          //realCodes(validationDoc.itemId) = Set(validationDoc.y.toArray:_*)
          realCodes(validationDoc.itemId) = Set(Codes.fromString("country").toArray:_*).intersect(validationDoc.y)
        }
        assignedCodes(validationDoc.itemId) ++= //adds the codes
          (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(validationDoc.x)), code)
          }.filter(_._1 > myCut).map(_._2).toSet)
      }

    }
      var buf = scala.collection.mutable.ListBuffer.empty[Tuple2[Set[String], Set[String]]]
      assignedCodes.foreach {
        case (itemid, assigned) =>
        buf += new Tuple2(assignedCodes(itemid) , realCodes(itemid))
          println(s"${assignedCodes(itemid)}" + s"${realCodes(itemid)}")
      }

      logger.log(s"average codes assigned per doc in total: ${1.0 * assignedCodes.map(_._2.size).sum / assignedCodes.size}")


      val validationPrecisionRecall = buf.map { case (actual, expected) =>
        (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
          actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
      }

      logger.log("Computing score")
      logger.log(s"average precision : ${validationPrecisionRecall.map(_._1).sum / 10000.0}")
      logger.log(s"average recall   :  ${validationPrecisionRecall.map(_._2).sum / 10000.0}")
      val validationF1 = validationPrecisionRecall
        .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue)}
      logger.log(s"score : ${validationF1.sum / 10000.0}")
    return (validationF1.sum / 10000.0)
  }

  def predict(cutoff:Double) : Unit = {
      var assignedCodes: Map[Int,Set[String]] = Map()
      //assign codes
      for (labelType <- labelTypes) {
        for (testDoc <- reader.toBagOfWords("test")) {
          if (!assignedCodes.contains(testDoc.itemId)) {
            assignedCodes(testDoc.itemId) = Set()
          }
          assignedCodes(testDoc.itemId) ++= //adds the codes
            (thetasMap(labelType).map { case (code, theta) => (logistic(theta.dot(testDoc.x)), code)
            }.filter(_._1 > cutoff).map(_._2).toSet)
        }

      }

      import java.io.PrintWriter
      import java.io.File
      val pw = new PrintWriter(new File("countrycodes.txt"))
      assignedCodes.toSeq.sortBy(_._1).foreach{ case(id, codes) =>
        var line = s"$id "
        codes.foreach(x=> line = line.concat(x + " "))
        pw.write(line + "\r\n")
      }
      pw.close()
    }

}