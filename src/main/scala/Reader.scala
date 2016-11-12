import java.io.File

import com.github.aztek.porterstemmer.PorterStemmer
//import scala.util.{Try, Success, Failure

import breeze.linalg.{DenseVector, SparseVector, Vector, VectorBuilder}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.processing.StopWords.stopWords

/**
  * Created by marc on 31/10/16.
  */


/**
  * Represents one data item in our project.i
  * Consints of x, the bag-of-words-vector and y, the set of codes.
  * @param x Bag of Words
  * @param y Lables
  */
case class DataPoint(itemid: Int, x: SparseVector[Double], y: Set[String])

/**
  * Reads all samples from the training data and forms a dictionary from that.
  * Provides methods to read other data sets as streams of bag-of-words-vectors
  * (based on the trained dictionary)
  *
  * @param minOccurrence     Minimum number of documents that a word must be included in.
  * @param maxOccurrenceRate Words that are in more than maxOccuranceRate*nrDocuments documents
  *                         are discared. Should be in (0, 1].
  * @param bias             Indicates wheter to include an extra 1 in bag-of-words vectors
  */
class Reader(minOccurrence: Int = 1,
             maxOccurrenceRate: Double = 0.2,
             bias: Boolean = true) {

  def tokenToWord(token: String) = PorterStemmer.stem(token.toLowerCase)

  val pattern = "[\\p{L}\\-]+".r.pattern
  def filterWords(word: String) = !stopWords.contains(word) && pattern.matcher(word).matches()



  println(" --- READER : Initializing Stream.")
  private val r = new ReutersRCVStream(new File("./src/main/resources/data/train").getCanonicalPath, ".zip")
  private val wordCounts = scala.collection.mutable.HashMap[String, Int]()
  println(" --- READER : Counting total number of documents.")
  private val docCount = r.length
  println(s" --- READER :    => got $docCount")
  var codes = scala.collection.mutable.Set[String]()
  private val numDocsPerCode = scala.collection.mutable.HashMap[String, Int]()


  println(" --- READER : Counting word- and code-occurences in training corpus...")
  //count words in files
  for (doc <- r.stream) {
    doc.tokens.distinct.map(tokenToWord).filter(filterWords).distinct.foreach(x => wordCounts(x) = 1 + wordCounts.getOrElse(x, 0))
    doc.codes.foreach(codes += _)
    doc.codes.foreach(x => numDocsPerCode(x) = 1 + numDocsPerCode.getOrElse(x,0))
  }

  codes = codes.intersect(possibleCodes.topicCodes)


  println(" --- READER : Filtering out words...")
  private val acceptableCount = maxOccurrenceRate * docCount
  val originalDictionarySize = wordCounts.size
  //compute dictionary (remove unusable words)
  val dictionary = wordCounts.filter(x => !stopWords.contains(x._1) && x._2 <= acceptableCount && x._2 >= minOccurrence)
    .keys.toList.sorted.zipWithIndex.toMap
  val reducedDictionarySize = dictionary.size
  println(s" --- READER :     => reduced dictionary size from $originalDictionarySize to $reducedDictionarySize")
  println(s" --- READER :     => Total number of considered codes : ${codes.size}")
  val outLength = reducedDictionarySize + (if (bias) 1 else 0)

  /**
    * Load the datapoints for a given collection.
    * @param collectionName The name of the collection. Either "test", "train" or "validation"
    * @return Stream of datapoints.
    */
  def toBagOfWords(collectionName: String): Stream[DataPoint] = {
    toBagOfWords(new ReutersRCVStream(
      new File("./src/main/resources/data/" + collectionName).getCanonicalPath, ".zip").stream)
  }
  /**
    * Load the datapoints for a given stream of documents.
    * @param input The documents.
    * @return Stream of datapoints.
    */
  def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] =
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      doc.tokens.map(tokenToWord).filter(filterWords).groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key)) (dictionary(key), count) else (-1, 0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
        DataPoint(doc.ID, v.toSparseVector(true, true), doc.codes.intersect(codes))
    })

  /**
    * Calculates probability of a code (#docs which contain code / total nb of docs)
    * @param code The code of a class
    * @return The probability of the code in the training set
   */
  def getProbabilityOfCode(code: String): Double = {
    val numDocs = numDocsPerCode.get(code)
    numDocs match{
      case Some(numDocs) => numDocs.toDouble / docCount.toDouble
      case None => 0.0
    }
  }

}
