/**
  * Created by Philip on 31/10/16.
  */

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer
import scala.io.Source
import scala.annotation.tailrec



/**
  *
  */
object NaiveBayes{

  val topicThreshold:Double = -10.7
  val countryThreshold:Double = -10.7
  val industryThreshold:Double = -10.7
  val logger = new Logger("NaiveBayes")
  var documentCategoryProbabilities = scala.collection.mutable.HashMap.empty[String,DenseVector[Double]].par
  //val reader = new Reader(10, 0.8, false) //create reader with no bias
  val reader = new TfIDfReader(2000, false)
  reader.pruneRareCodes(10)
  val topicCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("topic"))
  val industryCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("industry"))
  val countryCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("country"))


  /**
    *
    * @param wordCounts
    * @param alpha
    * @param code
    * @param bowStream
    * @param vocabSize
    * @return
    */
  @tailrec
  def calculateWordCategoryProbabilities(wordCounts: DenseVector[Double], alpha: Double, code: String, bowStream: Stream[DataPoint], vocabSize: Double):
      DenseVector[Double] = {
    if (bowStream.isEmpty) {
      log(wordCounts) - log(sum(wordCounts) + alpha*vocabSize)
    }
    else{
      val doc = bowStream.head
      if(doc.y contains code){
        calculateWordCategoryProbabilities(wordCounts + doc.x, alpha, code, bowStream.tail, vocabSize)
      }else{
        calculateWordCategoryProbabilities(wordCounts, alpha, code, bowStream.tail, vocabSize)
      }

    }
  }

  /**
    *
    * @param filename
    */
  def saveDocumentCategoryProbabilitiesToFile(filename:String): Unit ={
    logger.log("Training done. Saving model...")
    val pw = new PrintWriter(new File(filename))
    documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) => wordCategoryProbabilities.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write(_))
    pw.close
  }

  /**
    *
    * @param filename
    */
  def loadDocumentCategoryProbabilitiesFromFile(filename:String): Unit ={
    logger.log("Loading saved model from file")
    documentCategoryProbabilities.clear()
    for (line <- Source.fromFile(filename).getLines()) {
      val splitting = line.split("\t")
      val code = splitting(0)
      val wordCategoryProbabilities = splitting.slice(1,splitting.length).map(_.toDouble)
      documentCategoryProbabilities += code -> DenseVector[Double](wordCategoryProbabilities)
    }
  }

  /**
    *
    * @return
    */
  def getValidationResult(): List[(Set[String],Set[String])] ={
    reader.toBagOfWords("validation").map(dp =>
      (documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
        (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList.sortBy(_._1)
        .filter{ case (score, code) =>
                (score > topicThreshold && (topicCodes contains code)) || (score > industryThreshold && (industryCodes contains code)) || (score > countryThreshold && (countryCodes contains code))}
        .map(_._2)
        .toSet, dp.y)).toList
  }

  def validate(): Unit ={
    logger.log("Running trained model on validation data...")
    val validationResult = getValidationResult()

    //compute precision, recall, f1 and averaged f1
    logger.log("Computing scores")
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
    }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
    logger.log(" F1-Average=" + validationF1.sum / validationF1.length)
  }

  /**
    *
    * @param k
    */
  def predict(): Unit ={
    logger.log("Predicting codes for test-set")
    val testResult = reader.toBagOfWords("test").map(dp =>documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code, dp.itemId)}
      .toList
      .sortBy(_._1)
      .filter{ case (score, code, itemId) =>
        (score > topicThreshold && (topicCodes contains code)) || (score > industryThreshold && (industryCodes contains code)) || (score > countryThreshold && (countryCodes contains code))}
      .map(x => (x._2, x._3)).toSet)

    logger.log("Training done. Saving model...")
    val out = new PrintWriter(new File("./src/main/resources/data/ir-project-2016-1-7-nb.txt"))
    documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) => wordCategoryProbabilities.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(out.write(_))
    out.close

  }

  /**
    *
    */
  def train(): Unit ={
    logger.log("Training Model...")
    val codes = Set[String](reader.codes.toList: _*)
    logger.log(s"${codes.size} codes will be trained...")
    codes.foreach(code => documentCategoryProbabilities += code -> DenseVector.ones[Double](reader.reducedDictionarySize))
    var codesDone = Set[String]()
    documentCategoryProbabilities = documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      logger.log(codesDone.size + " codes passed", "trainI", 5)
      codesDone += code
      code -> calculateWordCategoryProbabilities(wordCounts = wordCategoryProbabilities,
        alpha = 1.0,
        code = code,
        bowStream = reader.toBagOfWords("train"),
        vocabSize = reader.reducedDictionarySize)
    }
  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    train()
    saveDocumentCategoryProbabilitiesToFile("./src/main/resources/data/model/bayesPar_2000tfidf_complete.csv")
    //loadDocumentCategoryProbabilitiesFromFile("./src/main/resources/data/model/bayesPar_2_0.2_stemmed.csv")
    validate()

  }
}