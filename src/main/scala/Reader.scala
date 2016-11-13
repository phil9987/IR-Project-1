import java.io.File

import com.github.aztek.porterstemmer.PorterStemmer
import breeze.linalg.{DenseVector, SparseVector, Vector, VectorBuilder}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.processing.StopWords.stopWords

/**
  * Created by marc on 31/10/16.
  */

/**
  * Represents one data item in our project.i
  * Consints of x, the bag-of-words-vector and y, the set of codes and the id of the document.
  * @param itemId The Id of the document
  * @param x      Bag of Words
  * @param y      Lables
  */
case class DataPoint(itemId: Int, x: SparseVector[Double], y: Set[String])

/**
  * Base class for the reader.
  * Implements structure and functions common to all implementation of Reader.
  */
abstract class BaseReader()
{
  protected val logger = new Logger("Reader")
  //cache used by cachedStem
  protected val stemmingCache = scala.collection.mutable.HashMap[String, String]()

  /**
    * Translate a token to its stemmed base word.
    * Uses a cache (HashMap) in order not to duplicate calculations.
    * @param token
    * @return stemmed word.
    */
  def cachedStem(token: String) = {
    if (!stemmingCache.contains(token)) stemmingCache(token) = PorterStemmer.stem(token.toLowerCase)
    stemmingCache(token)
  }

  /**
    * Given a token, transform it to a word.
    * @param token
    * @return word
    */
  def tokenToWord(token: String) = cachedStem(token)

  //pattern used by filterWords: letter or '-'
  protected val pattern = "[\\p{L}\\-]+".r.pattern

  /**
    * Filters words.
    * Current implementation removed stopwords and words not made up from letters and '-'.
    * @param word
    * @return Boolean indicating wheter to remove the word or not.
    */
  def filterWords(word: String) = !stopWords.contains(word) && pattern.matcher(word).matches()
}

/**
  * Reads all samples from the training data and forms a dictionary from that.
  * Provides methods to read other data sets as streams of bag-of-words-vectors
  * (based on the trained dictionary)
  *
  * @param minOccurrence     Minimum number of documents that a word must be included in.
  * @param maxOccurrenceRate Words that are in more than maxOccurrenceRate*nrDocuments documents
  *                         are discarded. Should be in (0, 1].
  * @param bias             Indicates whether to include an extra 1 in bag-of-words vectors
  */
class Reader(minOccurrence: Int = 1,
             maxOccurrenceRate: Double = 0.2,
             bias: Boolean = true) extends BaseReader {


  logger.log("Initializing Stream.")
  private val r = new ReutersRCVStream(new File("./src/main/resources/data/train").getCanonicalPath, ".zip")
  private val wordCounts = scala.collection.mutable.HashMap[String, Int]()
  logger.log("Counting total number of documents.")
  private val docCount = r.length
  logger.log(s"=> got $docCount")
  var codes = scala.collection.mutable.Set[String]()
  private val numDocsPerCode = scala.collection.mutable.HashMap[String, Int]()


  logger.log("Counting word- and code-occurences in training corpus...")
  //count words in files
  for (doc <- r.stream) {
    doc.tokens.distinct.map(tokenToWord).filter(filterWords).distinct.foreach(x => wordCounts(x) = 1 + wordCounts.getOrElse(x, 0))
    doc.codes.foreach(codes += _)
    doc.codes.foreach(x => numDocsPerCode(x) = 1 + numDocsPerCode.getOrElse(x,0))
  }


  logger.log("Filtering out words...")
  private val acceptableCount = maxOccurrenceRate * docCount
  val originalDictionarySize = wordCounts.size
  //compute dictionary (remove unusable words)
  val dictionary = wordCounts.filter(x => x._2 <= acceptableCount && x._2 >= minOccurrence)
    .keys.toList.sorted.zipWithIndex.toMap
  val reducedDictionarySize = dictionary.size
  logger.log(s"=> reduced dictionary size from $originalDictionarySize to $reducedDictionarySize")
  logger.log(s"=> Total number of considered codes : ${codes.size}")
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
        DataPoint(doc.ID, v.toSparseVector(true, true), doc.codes)
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
