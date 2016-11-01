
import ch.ethz.dal.tinyir.processing._
import breeze.linalg.{Vector, DenseVector}



object Main{
  def main(args : Array[String]): Unit ={

    val r = new Reader(2, 0.2, true)
    println("Read %d words from training set. Reduced to %d words.".format(r.originalDictionarySize, r
      .reducedDictionarySize))

    println(r.toBagOfWords("train").take(2))
    println(r.toBagOfWords("test").take(2))
    println(r.toBagOfWords("validation").take(2))

//    var testCorpus = new ReutersCorpusIterator("test")
//    for( i <- 1 until 10){
//      val doc : ReutersRCVParse = testCorpus.next()
//      println(doc.title)
//    }
  }
}