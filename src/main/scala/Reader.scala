import java.io.File

import com.github.aztek.porterstemmer.PorterStemmer
import breeze.linalg.{DenseVector, SparseVector, Vector, VectorBuilder}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.{Tokenizer, XMLDocument}
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
  private val logger = new Logger("BaseReader")

  //fast version of sort().take(n)
  //based on
  //https://stackoverflow.com/questions/8274726/top-n-items-in-a-list-including-duplicates
  def topNs(xs: TraversableOnce[(String,Int)], n: Int) = {
    var ss = List[(String,Int)]()
    var max = Int.MinValue
    var len = 0
    xs foreach { e =>
      if (len < n || e._2 < max) {
        ss = (e :: ss).sorted
        max = ss.head._2
        len += 1
      }
      if (len > n) {
        ss = ss.tail
        max = ss.head._2
        len -= 1
      }
    }
    ss
  }



  //cache used by cachedStem
  protected val stemmingCache = scala.collection.mutable.HashMap[String, String]()

  /**
    * Translate a token to its stemmed base word.
    * Uses a cache (HashMap) in order not to duplicate calculations.
    * @param token The token to be stemmed
    * @return stemmed word.
    */
  def cachedStem(token: String) = {
    if (!stemmingCache.contains(token)) stemmingCache(token) = PorterStemmer.stem(token.toLowerCase)
    stemmingCache(token)
  }

  /**
    * Given a token, transform it to a word. Currently only stemming.
    * @param token The token to be transformed.
    * @return word THe processed word.
    */
  def tokenToWord(token: String) = cachedStem(token)

  //pattern used by filterWords: letter or '-'
  protected val pattern = "[\\p{L}\\-]+".r.pattern

  /**
    * Filters words.
    * Current implementation removes stopwords and words not made up from letters and '-'.
    * @param word The word to be filtered.
    * @return Boolean indicating whether to remove the word or not.
    */
  def filterWords(word: String) = !stopWords.contains(word) && pattern.matcher(word).matches()


  protected val wordCounts = scala.collection.mutable.HashMap[String, Int]()
  var docCount = 0
  protected val numDocsPerCode = scala.collection.mutable.HashMap[String, Int]()
  var codes = scala.collection.mutable.Set[String]()

  protected def init() = {
    logger.log("init")
    logger.log("init: Initializing Stream.")
    val r = new ReutersRCVStream(new File("./src/main/resources/data/train").getCanonicalPath, ".zip")
    docCount = r.length
    logger.log("init: Counting word- and code-occurences in training corpus...")
    for (doc <- r.stream) {
      doc.tokens.distinct.map(tokenToWord).filter(filterWords).distinct.foreach(x => wordCounts(x) = 1 + wordCounts.getOrElse(x, 0))
      doc.codes.foreach(codes += _)
      doc.codes.foreach(x => numDocsPerCode(x) = 1 + numDocsPerCode.getOrElse(x,0))
    }

  }

  init()

  def pruneRareCodes(k: Int): Unit ={
    logger.log(s"number of codes before pruning rare ones: ${codes.size}")
    codes = codes -- Set[String](numDocsPerCode.filter(x => x._2 < 10).keys.toList: _*)
    logger.log(s"number of codes after pruning rare ones: ${codes.size}")
  }


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
  {
    List[DataPoint]().toStream
  }

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
  protected val logger = new Logger("Reader")
  logger.log(s"Starting Reader minOccurrence=$minOccurrence, maxOccurrenceRate=$maxOccurrenceRate, bias=$bias")

  logger.log("Filtering out words...")
  protected val acceptableCount = maxOccurrenceRate * docCount
  val originalDictionarySize = wordCounts.size
  //compute dictionary (remove unusable words)
  val dictionary = wordCounts.filter(x => x._2 <= acceptableCount && x._2 >= minOccurrence)
    .keys.toList.sorted.zipWithIndex.toMap
  val reducedDictionarySize = dictionary.size
  logger.log(s"=> reduced dictionary size from $originalDictionarySize to $reducedDictionarySize")
  logger.log(s"=> Total number of considered codes : ${codes.size}")
  val outLength = reducedDictionarySize + (if (bias) 1 else 0)

  /**
    * Load the datapoints for a given stream of documents.
    *
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      doc.tokens.map(tokenToWord).filter(filterWords).groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key)) (dictionary(key), count) else (-1, 0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
      DataPoint(doc.ID, v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true), doc.codes)
    })
  }
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
class ReaderTfIdfWeighted(minOccurrence: Int = 1,
             maxOccurrenceRate: Double = 0.2,
             bias: Boolean = true) extends BaseReader {
  protected val logger = new Logger("Reader")
  logger.log(s"Starting Reader minOccurrence=$minOccurrence, maxOccurrenceRate=$maxOccurrenceRate, bias=$bias")

  logger.log("Calculating idf")
  val idf = wordCounts.toList.par.map(x => (x._1, Math.log( docCount.toDouble/x._2
    .toDouble
  ))).toMap.seq
  logger.log("Filtering out words...")
  protected val acceptableCount = maxOccurrenceRate * docCount
  val originalDictionarySize = wordCounts.size
  //compute dictionary (remove unusable words)
  val dictionary = wordCounts.filter(x => x._2 <= acceptableCount && x._2 >= minOccurrence)
    .keys.toList.sorted.zipWithIndex.toMap
  val reducedDictionarySize = dictionary.size
  logger.log(s"=> reduced dictionary size from $originalDictionarySize to $reducedDictionarySize")
  logger.log(s"=> Total number of considered codes : ${codes.size}")
  val outLength = reducedDictionarySize + (if (bias) 1 else 0)

  /**
    * Load the datapoints for a given stream of documents.
    *
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      doc.tokens.map(tokenToWord).filter(filterWords).groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key) && idf.contains(key)) (dictionary(key), count.toDouble*idf(key)) else (-1, 0.0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
      DataPoint(doc.ID, breeze.linalg.normalize(v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true)), doc.codes)
    })
  }
}

/**
  * Ranks the words according to their idf, keeping only the top N.
  * Computes the bag-of-words representations for documents based on those
  * and weights them using tf-idf.
  * @param topNDocs The number of words to remain in the dictionary (sorted by idf).
  * @param bias     Indicates whether to include an extra 1 in bag-of-words vectors
  */
class TfIDfReader(topNDocs: Int, bias: Boolean = true) extends BaseReader {
  private val logger = new Logger("TfIDfReader")
  logger.log(s"Starting TfIDfReader topN=$topNDocs bias=$bias")

  logger.log("Finding top documents...")
  var top = wordCounts.toList
  if (topNDocs > 0)
    top =  topNs(wordCounts.toList,topNDocs)
  logger.log("Calculating idf")
  val idf = top.par.map(x => (x._1, Math.log( docCount.toDouble/x._2
    .toDouble
  ))).toMap.seq
  val dictionary = idf.keys.toList.sorted.zipWithIndex.toMap
  val outLength = dictionary.size + (if (bias) 1 else 0)
  logger.log(s"Dictionary size = $outLength")

  /**
    * Load the datapoints for a given stream of documents.
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override   def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      val actualWords = doc.tokens.map(tokenToWord).filter(filterWords).filter(dictionary.contains)
      actualWords.groupBy(identity).mapValues(_.size).toList
        .filter(x => dictionary.contains(x._1))
        .map {
               case (key, count) => (dictionary(key), idf(key) * count.toDouble / actualWords.length)
             }.sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(outLength - 1, 1) //bias
      val bagOfWordsVector = v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true)
      DataPoint(doc.ID, breeze.linalg.normalize(bagOfWordsVector), doc.codes.intersect(codes))
    })
  }
}


/**
  * Trait that changes the BaseReader only to use tokens from the Title of the document.
  */
trait TitleReaderSetup extends BaseReader {
  protected override def init() = {
    val r = new ReutersRCVStream(new File("./src/main/resources/data/train").getCanonicalPath, ".zip")
    docCount = r.length
    for (doc <- r.stream) {
      Tokenizer.tokenize(doc.title).distinct.map(tokenToWord).filter(filterWords).distinct.foreach(x => wordCounts(x) = 1 + wordCounts.getOrElse(x, 0))
      doc.codes.foreach(codes += _)
      doc.codes.foreach(x => numDocsPerCode(x) = 1 + numDocsPerCode.getOrElse(x,0))
    }
  }
}


//Same as Reader, but only using titles
class TitleReader(minOccurrence: Int = 1,
                  maxOccurrenceRate: Double = 0.2,
                  bias: Boolean = true) extends Reader(minOccurrence, maxOccurrenceRate, bias) with TitleReaderSetup{

  /**
    * Load the datapoints for a given stream of documents.
    *
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      Tokenizer.tokenize(doc.title).map(tokenToWord).filter(filterWords).groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key)) (dictionary(key), count) else (-1, 0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
      DataPoint(doc.ID, v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true), doc.codes)
    })
  }
}

//Same as ReaderTfIdfWeighted, but only using titles
class TitleReaderTfIdfWeighted(minOccurrence: Int = 1,
                               maxOccurrenceRate: Double = 0.2,
                               bias: Boolean = true) extends ReaderTfIdfWeighted(minOccurrence, maxOccurrenceRate,
                                                                                 bias) with TitleReaderSetup{
  /**
    * Load the datapoints for a given stream of documents.
    *
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      Tokenizer.tokenize(doc.title).map(tokenToWord).filter(filterWords).groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key) && idf.contains(key)) (dictionary(key), count.toDouble*idf(key)) else (-1, 0.0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
      DataPoint(doc.ID,  v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true), doc.codes)
    })
  }
}

//Same as TfIDfTitleReader, but only using titles
class TfIDfTitleReader(topNDocs: Int, bias: Boolean = true) extends TfIDfReader(topNDocs, bias)  with TitleReaderSetup{
  private val logger = new Logger("TfIDfReader")
  logger.log(s"Starting TfIDfReader topN=$topNDocs bias=$bias")

  /**
    * Load the datapoints for a given stream of documents.
    * @param input The documents.
    * @return Stream of datapoints.
    */
  override   def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] = {
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      val actualWords = Tokenizer.tokenize(doc.title).map(tokenToWord).filter(filterWords).filter(dictionary.contains)
      actualWords.groupBy(identity).mapValues(_.size).toList
        .filter(x => dictionary.contains(x._1))
        .map {
               case (key, count) => (dictionary(key), idf(key) * count.toDouble / actualWords.length)
             }.sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(outLength - 1, 1) //bias
      val bagOfWordsVector = v.toSparseVector(alreadySorted = true, keysAlreadyUnique = true)
      DataPoint(doc.ID, breeze.linalg.normalize(bagOfWordsVector), doc.codes.intersect(codes))
    })
  }
}
