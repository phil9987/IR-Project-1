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
  * Creates a Naive Bayes classifier on the reuters dataset with the given thresholds
  * for topic- and country-codes while not classifying the industry-codes.
  * @param topicThreshold
  * @param countryThreshold
  */
class NaiveBayes(topicThreshold: Double, countryThreshold:Double){

  // the logger we will use to print infos & progress
  private val logger = new Logger("NaiveBayes")

  // the probabilities P(document | category), represented as a Hashmap mapping codes to bag-of-word-vectors
  private var documentCategoryProbabilities = scala.collection.mutable.HashMap.empty[String,DenseVector[Double]].par

  // our reader for the topic-codes, minimum-occurrence=3, maximum-occurrence=0.2*docCount, no bias
  private val reader = new Reader(3, 0.2, false)

  // our reader for the country-codes, minimum-occurrence=20, no maximum-occurrence, no bias
  private val titleReader = new TitleReader(20, 1, false)

  // set of all topic-codes
  private val topicCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("topic"))

  // set of all country-codes
  private val countryCodes = Set[String](reader.codes.toList: _*).intersect(Codes.fromString("country"))

  /**
    * Calculates the probability P(w | c)
    * @param wordCounts bag-of-words vector of the dictionary, initially all 1
    * @param alpha smoothing parameter (default: 1 for Laplace-smoothing)
    * @param code  current class-code to be calculated
    * @param bowStream stream of training documents
    * @param vocabSize the size of the full bag-of-words vector
    * @return log(P(w | c)) where w is a bag-of-words-vector consisting of all words in the reader dictionary
    */
  @tailrec
  private def calculateWordCategoryProbabilities(wordCounts: DenseVector[Double],
                                                 alpha: Double,
                                                 code: String,
                                                 bowStream: Stream[DataPoint],
                                                 vocabSize: Double): DenseVector[Double] = {
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
    * Saves the trained DocumentCategoryProbabilities to filename
    * @param filename the full path to the file
    */
  def saveDocumentCategoryProbabilitiesToFile(filename:String): Unit ={
    logger.log("Training done. Saving model...")
    val pw = new PrintWriter(new File(filename))
    documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      wordCategoryProbabilities.toArray.mkString(code + "\t", "\t", "\n") }
      .seq.foreach(pw.write(_))
    pw.close
  }

  /**
    * Loads the trained DocumentCategoryProbabilities from filename (full path)
    * @param filename full path to saved DocumentCategoryProbabilitiesFile
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
    * Trains the model on the trainingdata, using a different reader for the
    * topic-codes and the country-codes (industry-codes are not trained)
    */
  def train(): Unit ={
    logger.log("Training Model...")
    var codes = Set[String](reader.codes.toList: _*).intersect(topicCodes)

    logger.log(s"${codes.size} codes will be trained...")
    codes.foreach(code =>
      documentCategoryProbabilities += (code -> DenseVector.ones[Double](reader.reducedDictionarySize) ))
    var codesDone = Set[String]()
    documentCategoryProbabilities = documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      logger.log(codesDone.size + " codes passed", "trainI", 5)
      codesDone += code
      (code -> calculateWordCategoryProbabilities(wordCounts = wordCategoryProbabilities,
                                                  alpha = 1.0,
                                                  code = code,
                                                  bowStream = reader.toBagOfWords("train"),
                                                  vocabSize = reader.reducedDictionarySize))
    }

    // train countryCodes with TitleReader..
    codes = Set[String](titleReader.codes.toList: _*).intersect(countryCodes)   //get country codes
    logger.log(s"${codes.size} codes will be trained...")
    codes.foreach(code =>
      documentCategoryProbabilities += (code -> DenseVector.ones[Double](titleReader.reducedDictionarySize) ))
    codesDone = Set[String]()     // for logging
    val documentCategoryProbabilitiesCountry = documentCategoryProbabilities.filter(countryCodes contains _._1)
      .map { case (code, wordCategoryProbabilities) =>
      logger.log(codesDone.size + " codes passed", "trainI", 5)
      codesDone += code
      code -> calculateWordCategoryProbabilities( wordCounts = wordCategoryProbabilities,
                                                  alpha = 1.0,
                                                  code = code,
                                                  bowStream = titleReader.toBagOfWords("train"),
                                                  vocabSize = titleReader.reducedDictionarySize)
    }
    // merge documentCategoryProbabilitiesCountry into documentCategoryProbabilities
    documentCategoryProbabilitiesCountry.foreach(dp => documentCategoryProbabilities += dp)
  }

  /**
    * Calculates the predictions on the validation set
    * @return a list of (predicted, expected) tuples
    */
  def getValidationResult(): List[(Set[String],Set[String])] ={
    var topicCodesResult = reader.toBagOfWords("validation").map(dp =>
      (documentCategoryProbabilities.filter(topicCodes contains _._1)
        .map { case (code, wordCategoryProbabilities) =>
          (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList.sortBy(_._1)
        .filter{ case (score,code)=> (score > topicThreshold)}
        .map(_._2).toSet, dp.y)).toList

    var countryCodesResult = titleReader.toBagOfWords("validation").map(dp =>
      (documentCategoryProbabilities.filter(countryCodes contains _._1)
        .map { case (code, wordCategoryProbabilities) =>
          (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList.sortBy(_._1)
        .filter{ case (score,code)=> (score > countryThreshold)}
        .map(_._2).toSet)).toList

    // return: merged results of country- and topic-codes
    topicCodesResult.zip(countryCodesResult).map{ case ((topicSet, expectedSet), countrySet) =>
      (topicSet.union(countrySet), expectedSet)}.toList
  }

  /**
    * Calculates the performance on the validation set and prints the result
    * The thresholds used are defined as global variables 'topicThreshold' and
    * 'countryThreshold'
    */
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
        .map { case (precision, recall) =>
          (2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue)) }
      logger.log( s"Topic-Threshold = $topicThreshold, " +
                  s"Country-Threshold = $countryThreshold, " +
                  s"industryThreshold = 0.0, " +
                  s"F1-Average= ${validationF1.sum / validationF1.length}")
  }

  /**
    * Predicts the codes for the test-set and stores the result in the format
    * docId code1 code2 code3 ... (separated by whitespace, each doc on new line)
    * The thresholds used are defined as global variables 'topicThreshold' and
    * 'countryThreshold'
    * @param k
    */
  def predict(filename: String): Unit ={
    logger.log("Predicting codes for test-set")
    val topicCodesTestResult = reader.toBagOfWords("test").map(dp =>
      (dp.itemId, documentCategoryProbabilities.filter(topicCodes contains _._1)
                    .map { case (code, wordCategoryProbabilities) =>
                      (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList
        .filter{ case (score, code) => (score > topicThreshold)}
        .map(_._2).toSet)).toList

    val countryCodesTestResult = titleReader.toBagOfWords("test").map(dp =>
      (documentCategoryProbabilities.filter(countryCodes contains _._1)
        .map { case (code, wordCategoryProbabilities) =>
          (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)}
        .toList
        .filter{ case (score, code) => (score > countryThreshold)}
        .map(_._2).toSet)).toList

    val testResult = topicCodesTestResult.zip(countryCodesTestResult)
      .map{ case ((docId, topicSet), countrySet) =>
        (docId, topicSet.union(countrySet))}.toList
    logger.log("Prediction done. Saving test-output...")

    val out = new PrintWriter(new File(filename))
    testResult.map { case (docId, predictedCodes) =>
      predictedCodes.toArray.mkString(docId + " ", " ", "\n") }.seq.foreach(out.write(_))
    out.close
  }
}

/**
  * Companion Object with main method.
  * Used for internal testing of SVM.
  */
object NaiveBayes {
  def main(args: Array[String]): Unit = {
    val nb = new NaiveBayes(-10.2, -9.4)
    nb.train()
    nb.saveDocumentCategoryProbabilitiesToFile("NBCompleteModel")
    //nb.loadDocumentCategoryProbabilitiesFromFile("NBCompleteModel")
    nb.validate()
    //nb.predict("ir-2016-1-project-7-nb.txt")
  }
}