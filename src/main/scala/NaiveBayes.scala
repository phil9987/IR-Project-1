/**
  * Created by Philip on 31/10/16.
  */

import breeze.linalg._
import breeze.numerics.{log}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer


object NaiveBayes{
  //val numTrain = 500
  //val numTest = 1000

  /**
    *P(w|c)
    * @param theta
    * @param alpha
    * @param code
    * @param bowStream
    * @Param wordCountVector
    * @return The probability of the code in the training set
    */
  def getPwc(theta: DenseVector[Double], alpha: Double, code: String, bowStream: Stream[DataPoint], wordCountVector: SparseVector[Double]):
      SparseVector[Double] = {
    if (bowStream.isEmpty) theta / (sum(theta) + alpha*sum(wordCountVector))
    else{
      val doc = bowStream.head
      if(doc.y contains code){
        getPwc(theta + doc.x + (alpha)*DenseVector.ones[Double](theta.length), alpha, code, bowStream.tail,wordCountVector)
      }else{
        getPwc(theta, alpha, code, bowStream.tail, wordCountVector)
      }

    }
  }

  def main(args: Array[String]): Unit = {
    val r = new Reader(2, 0.7, false)   //create reader with no bias
    println(" --- NAIVE BAYES : Training Model...")
    val codes = scala.collection.immutable.Set[String](r.codes.toList: _*)
    var thetas = codes.map((_, DenseVector.ones[Double](r.reducedDictionarySize))).toMap.par
    //println(codes)
    //println(" --- NAIVE BAYES : P(CCAT) = " + r.getProbabilityOfCode("CCAT"))
    val theta = DenseVector.zeros[Double](r.reducedDictionarySize)
    val wordcounts = r.getWordCountVector()
    //println(wordcounts)
    //val pwc = getPwc(theta, 1.0, "CCAT", r.toBagOfWords("train"),wordcounts)
    //println(" --- NAIVE BAYES : Pwc calculated for CCAT")
    //println(pwc)
    println(" --- NAIVE BAYES : Calculating P(w|c) all codes in the training set...")
    thetas = thetas.map { case (code, theta) => code -> getPwc(theta = DenseVector.zeros[Double](r.reducedDictionarySize),
                                                                alpha = 1.0,
                                                                code = code,
                                                                bowStream = r.toBagOfWords("train"),
                                                                wordCountVector = wordcounts)
    }
    //println(thetas)

    //run on validation data
    println(" --- NAIVE BAYES : Running verification")
    println(thetas.map { case (code, theta) =>
      (log(theta), log(r.getProbabilityOfCode(code)))})
    val validationResult = r.toBagOfWords("validation").map(dp =>
      (thetas.map { case (code, theta) =>
        (log(r.getProbabilityOfCode(code)) + dp.x.dot(log(theta)), code)}
          .toList.sortBy(-_._1)
          .map(_._2)
        .toSet, dp.y)).toList
    println(validationResult)
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