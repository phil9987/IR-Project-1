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

  var topicThreshold:Double = -7.1
  var countryThreshold:Double = -7.4
  //var industryThreshold:Double = -10.3
  var industryThreshold:Double = 0
  val logger = new Logger("NaiveBayes")
  var documentCategoryProbabilities = scala.collection.mutable.HashMap.empty[String,DenseVector[Double]].par
  //val reader = new TitleReader(20,1,false)
  val reader = new Reader(3, 0.2, false) //create reader with no bias
  //val titleReader = new TitleReader(20,1,false)
  //val reader = new TfIDfReader(2000, false)
  //val reader = new ReaderTfIdfWeighted(3, 0.8, false)
  //reader.pruneRareCodes(10)
  val topicCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("topic"))
  val industryCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("industry"))
  val countryCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("country"))
  val debugCodes = Set[String]("GPOL", "GPRO", "GREL", "GSCI", "GSPO", "GTOUR", "GVIO", "GVOTE", "GWEA", "GWELF", "M11", "M12", "M13")
  val thresholdPerCode = scala.collection.mutable.HashMap.empty[String,Double].par

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
  def saveThresholdPerCodeToFile(filename:String): Unit ={
    logger.log("saving ThresholdPerCode Hashmap to file...")
    val out1 = new PrintWriter(new File(filename))
    thresholdPerCode.map { case (code, threshold) => s"$code ${threshold.toString()}\n"}.seq.foreach(out1.write(_))
    out1.close()
  }

  def loadThresholdFromFile(filename:String): Unit ={
    logger.log("Loading saved thresholds per code from file")
    thresholdPerCode.clear()
    for (line <- Source.fromFile(filename).getLines()) {
      val splitting = line.split(" ")
      val code = splitting(0)
      thresholdPerCode += code -> splitting(1).toDouble
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
          .filter{ case (score,code)=>
            (score > thresholdPerCode.getOrElse(code, 0.0) && (topicCodes contains code)) || (score > industryThreshold && (industryCodes contains code)) || (score > countryThreshold && (countryCodes contains code))
          }
        .map(_._2).toSet, dp.y.intersect(topicCodes))).toList
  }

  /**
    *
    * @param code
    * @return
    */
  def getTrainValidationResult(code: String): List[(Set[String],Set[String])] ={
    reader.toBagOfWords("train").take(2000).map(dp =>
      (documentCategoryProbabilities.map { case (code1,wordCategoryProbabilities) =>
        (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
          .toList.filter{ case (score,code2)=> (code2.equals(code) && score > topicThreshold)}
          .map(_._2).toSet, if (dp.y contains code) Set[String](code) else Set[String]())).toList
  }

  /**
    *
    */
  def trainValidate(): Unit = {
    logger.log("Training optimal threshold for every code...")
    for (code <- topicCodes) {
      var threshold: Double = 0
      var lastScore: Double = -100.0
      var f1Average: Double = 0.0
      var step = 3.0
      var increasing = 1
      var stepsDone = 0
      var codesDone = 0
      while (increasing == 1 && stepsDone < 20) {
        topicThreshold = threshold
        val validationResult = getTrainValidationResult(code)
        logger.log("Computing scores")
        val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
          ((actual.intersect(expected).size.toDouble + scala.Double.MinPositiveValue) / (actual.size + scala.Double.MinPositiveValue),
            (actual.intersect(expected).size.toDouble + scala.Double.MinPositiveValue) / (expected.size + scala.Double.MinPositiveValue))
        }
        val validationF1 = validationPrecisionRecall.map { case (precision, recall)
        => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue)
        }
        f1Average = validationF1.sum / validationF1.length
        logger.log(s"Threshold = $topicThreshold F1-Average= $f1Average")
        if (f1Average > 0.9 && stepsDone > 5) {
          increasing = 0
        } else if (lastScore <= f1Average || f1Average == 0.0) {
            lastScore = f1Average
            threshold -= step
        } else if (step == 3.0) {
          // f1Average < lastScore -> revert threshold to better score and create smaller steps
          threshold += step
          step = 0.3
          threshold -= step
        } else {
          //step == 0.3 & f1Average < lastScore -> not increasing anymore!
          threshold += step
          increasing = 0
        }
        stepsDone += 1
      }
      thresholdPerCode += code -> threshold
      codesDone += 1
      logger.log(codesDone + " codes passed", "trainThresholds", 5)
    }

    saveThresholdPerCodeToFile("./src/main/resources/data/model/thresholdPerCodeTopics")

  }

  /**
    *
    */
  def validate(): Unit ={
    logger.log("Running trained model on validation data...")
    for(k<- 1 until 11 by 1) {
      topicThreshold = -9.5 -(0.1*k)
      val validationResult = getValidationResult()
      //compute precision, recall, f1 and averaged f1
      logger.log("Computing scores")
      val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
        (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
          actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
      }
      val validationF1 = validationPrecisionRecall
        .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
      logger.log(s"Threshold = $topicThreshold F1-Average= ${validationF1.sum / validationF1.length}")
    }
  }

  /**
    *
    * @param k
    */
  def predict(): Unit ={
    logger.log("Predicting codes for test-set")
    val testResult = reader.toBagOfWords("test").map(dp => (dp.itemId, documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
      .toList
      .filter{ case (score, code) =>
        (score > topicThreshold && (topicCodes contains code)) || (score > industryThreshold && (industryCodes contains code)) || (score > countryThreshold && (countryCodes contains code))}
        .map(_._2).toSet))
    logger.log("Prediction done. Saving model...")

    val out = new PrintWriter(new File("./src/main/resources/data/ir-project-2016-1-7-nb.txt"))
    testResult.map { case (docId, predictedCodes) => predictedCodes.toArray.mkString(docId + " ", " ", "\n") }.seq.foreach(out.write(_))
    out.close
  }

  /**
    *
    */
  def train(): Unit ={
    logger.log("Training Model...")
    //val codes = Set[String](reader.codes.toList: _*)
    //val codes = debugCodes
    //val codes = Set[String](reader.codes.toList: _*).intersect(countryCodes)
    var codes = Set[String](reader.codes.toList: _*).intersect(topicCodes)

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

    /*// train countryCodes with TitleReader...
    codes = Set[String](titleReader.codes.toList: _*).intersect(countryCodes)
    logger.log(s"${codes.size} codes will be trained...")
    codes.foreach(code => documentCategoryProbabilities += code -> DenseVector.ones[Double](reader.reducedDictionarySize))
    codesDone = Set[String]()
    documentCategoryProbabilities = documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      logger.log(codesDone.size + " codes passed", "trainI", 5)
      codesDone += code
      code -> calculateWordCategoryProbabilities(wordCounts = wordCategoryProbabilities,
        alpha = 1.0,
        code = code,
        bowStream = titleReader.toBagOfWords("train"),
        vocabSize = titleReader.reducedDictionarySize)
    }*/

  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    //train()
    //saveDocumentCategoryProbabilitiesToFile("./src/main/resources/data/model/bayesPar_3_0.2_topicCodes_stemmed.csv")
    loadDocumentCategoryProbabilitiesFromFile("./src/main/resources/data/model/bayesPar_3_0.2_topicCodes_stemmed.csv")
    //trainValidate()
    loadThresholdFromFile("./src/main/resources/data/model/thresholdPerCodeTopics")
    validate()
    //predict()

  }
}