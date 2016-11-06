/**
  * Created by Philip on 31/10/16.
  */

import breeze.linalg.{Vector, DenseVector}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer


object NaiveBayes{
  //val numTrain = 500
  //val numTest = 1000

  def main(args: Array[String]): Unit = {
    println("Reading Data...")
    val r = new Reader(0, 1, false)   //create reader with no bias
    println("Training Naive Bayes Model...")
    val codes = scala.collection.immutable.Set[String](r.codes.toList: _*)
    var thetas = codes.map((_, DenseVector.fill[Double](r.reducedDictionarySize + 1)(1.0))).toMap.par
    println("Theta size: " + thetas.size)

    println("P(CCAT) = " + r.getProbabilityOfCode("CCAT"))


    /*val resourceFolder = getClass.getResource("/data/").getPath
    val path = resourceFolder
    println(path)
    val reuters = new ReutersRCVStream (path + "train")
    println("Number of files in zips = " + reuters.length)

    val stream = reuters.stream
    val trainStream = stream.take(numTrain)
    val testStream = stream.drop(numTrain).take(numTest)
    val vocabSize = trainStream.flatMap(_.tokens).distinct.length
    println("vocabsize = " + vocabSize)

    val cat = "M13"         // category code MONEY_MARKETS
    val tks = trainStream.filter(_.codes(cat)).flatMap(_.tokens)
    val denominator = tks.length.toDouble + vocabSize
    val sum = tks.length.toDouble
    val PwcSparseNumerator = tks.groupBy(identity).mapValues(l => l.length.toDouble + 1.0)
    //println(PwcSparseNumerator)
    val word:String = "trigger"
    val prob = PwcSparseNumerator.getOrElse(word,1.0) / denominator
    println(prob)
    /* Library Example

    var length : Long = 0
    var tokens : Long = 0
    for (doc <- reuters.stream) {
      length += doc.content.length
      tokens += Tokenizer.tokenize(doc.content).length
    }
    println("Total number of characters = " + length)
    println("Total number of tokens     = " + tokens)*/*/


  }
}