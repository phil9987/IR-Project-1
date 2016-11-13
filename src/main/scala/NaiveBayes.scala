/**
  * Created by Philip on 31/10/16.
  */

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer
import scala.io.Source


/**
  *
  */
object NaiveBayes{

  val topicThreshold = -10.7
  val countryCodeThreshold = -10.7
  val industryCodeThreshold = -10.7
  val logger = new Logger("NaiveBayes")
  var documentCategoryProbabilities = scala.collection.mutable.HashMap.empty[String,DenseVector[Double]].par
  val reader = new Reader(2, 0.8, false) //create reader with no bias

  /**
    *
    * @param wordCounts
    * @param alpha
    * @param code
    * @param bowStream
    * @param vocabSize
    * @return
    */
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
    * @param k
    * @return
    */
  def getValidationResult(k: Double): List[(Set[String],Set[String])] ={
    val actual_codes = Set[String](reader.codes.toList:_*).intersect(Codes.fromString("topic"))
    reader.toBagOfWords("validation").map(dp =>
      (documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
        (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList.sortBy(_._1)
        .filter(_._1 > k)
        .map(_._2)
        .toSet.intersect(actual_codes), dp.y.intersect(actual_codes))).toList
  }

  def validate(): Unit ={
    logger.log("Running trained model on validation data...")
    for (k <- 1 until 10 by 1) {
      val validationResult = getValidationResult(-10-(k*0.1))


      //compute precision, recall, f1 and averaged f1
      logger.log("Computing scores")
      val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
        (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
          actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
      }
      val validationF1 = validationPrecisionRecall
        .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
      logger.log("k=" + (-10-(k*0.1)).toString() + " F1-Average=" + validationF1.sum / validationF1.length)
    }
  }

  /**
    *
    * @param k
    */
  def classifyTestSet(k: Double): Unit ={
    reader.toBagOfWords("test").map(dp =>documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
      .toList
      .sortBy(_._1)
      .filter(_._1 > k).map(_._2).toSet)
  }

  /**
    *
    */
  def train(): Unit ={
    logger.log("Training Model...")
    val topicCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("topic"))
    logger.log(topicCodes.size + " topic codes will be trained...")
    topicCodes.foreach(code => documentCategoryProbabilities += code -> DenseVector.ones[Double](reader.reducedDictionarySize))
    logger.log("Calculating P(d|c) for every (document, category) pair...")
    var codesDone = Set[String]()
    documentCategoryProbabilities = documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      logger.log(codesDone.size + " codes passed", "iPass", 5)
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
    saveDocumentCategoryProbabilitiesToFile("./src/main/resources/data/model/bayesPar_2_0.8_stemmed.csv")
    //loadDocumentCategoryProbabilitiesFromFile("./src/main/resources/data/model/bayesPar_2_0.2_stemmed.csv")
    validate()

  }
}