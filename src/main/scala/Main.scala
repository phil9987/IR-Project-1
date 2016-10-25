
import ch.ethz.dal.tinyir.processing._
import breeze.linalg.{Vector, DenseVector}



object Main{
  def main(args : Array[String]): Unit ={
    var testCorpus = new ReutersCorpusIterator("test")
    for( i <- 1 until 10){
      val doc : ReutersRCVParse = testCorpus.next()
      println(doc.title)
    }
  }
}