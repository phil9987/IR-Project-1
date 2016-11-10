/**
  * Created by Philip on 31/10/16.
  */

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer
import scala.io.Source



object NaiveBayes{

  var documentCategoryProbabilities = scala.collection.mutable.HashMap.empty[String,DenseVector[Double]].par
  val reader = new Reader(10, 0.8, false) //create reader with no bias

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
    println(" --- NAIVE BAYES : Training done. Saving model...")
    val pw = new PrintWriter(new File(filename))
    documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) => wordCategoryProbabilities.toArray.mkString(code + "\t", "\t", "\n") }.seq.foreach(pw.write(_))
    pw.close
  }

  /**
    *
    * @param filename
    */
  def loadDocumentCategoryProbabilitiesFromFile(filename:String): Unit ={
    println(" --- NAIVE BAYES : Loading saved model from file")
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
    reader.toBagOfWords("validation").map(dp =>
      (documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
        (log(reader.getProbabilityOfCode(code)) + dp.x.dot(wordCategoryProbabilities) / sum(dp.x), code)
      }
        .toList.sortBy(_._1)
        .filter(_._1 > k)
        .map(_._2)
        .toSet, dp.y)).toList
  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    val k = -9.9
    println(" --- NAIVE BAYES : Training Model...")
    /*val codes = reader.codes.toList
    println(" --- NAIVE BAYES : We will train " + codes.length + " codes")
    codes.foreach(code => documentCategoryProbabilities += code -> DenseVector.ones[Double](r.reducedDictionarySize))
    println(" --- NAIVE BAYES : Calculating P(d|c) for every document d, category c pair...")
    var i = 0
    documentCategoryProbabilities = documentCategoryProbabilities.map { case (code, wordCategoryProbabilities) =>
      i+=1
      if (i % 5 == 0) println(" --- NAIVE BAYES : " + i + " codes passed")
      code -> calculateWordCategoryProbabilities(wordCounts = wordCategoryProbabilities,
      alpha = 1.0,
      code = code,
      bowStream = r.toBagOfWords("train"),
      vocabSize = r.reducedDictionarySize)
    }

    saveDocumentCategoryProbabilitiesToFile("./src/main/resources/data/model/bayesPar3.csv")
*/ loadDocumentCategoryProbabilitiesFromFile("./src/main/resources/data/model/bayesPar_10_0.8.csv")




    //run on validation data
    println(" --- NAIVE BAYES : Running verification")
    val validationResult = getValidationResult(k)

    //compute precision, recall, f1 and averaged f1
    println(" --- NAIVE BAYES : Computing scores")
    val validationPrecisionRecall = validationResult.map { case (actual, expected) =>
      (actual.intersect(expected).size.toDouble / (actual.size + scala.Double.MinPositiveValue),
        actual.intersect(expected).size.toDouble / (expected.size + scala.Double.MinPositiveValue))
    }
    val validationF1 = validationPrecisionRecall
      .map { case (precision, recall) => 2 * precision * recall / (precision + recall + scala.Double.MinPositiveValue) }
    println(" --- NAIVE BAYES : k=" + k + " F1-Average=" + validationF1.sum / validationF1.length)
  }
}