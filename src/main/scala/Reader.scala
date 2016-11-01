import java.io.File

import breeze.linalg.{SparseVector, Vector, VectorBuilder}
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.XMLDocument

/**
  * Created by marc on 31/10/16.
  */


/**
  * Represents one data item in our project.
  * Consints of x, the bag-of-words-vector and y, the set of codes.
  * @param x Bag of Words
  * @param y Lables
  */
case class DataPoint(x: SparseVector[Double], y: Set[String])

/**
  * Reads all samples from the trainin gdata and form a dictionary from that.
  * Provides methods to read other data sets as streams of bag-of-words-vectors
  * (based on the trained dictionary)
  *
  * @param minOccurrence     Minimum number of documents that a word must be included in.
  * @param maxOccurrenceRate Words that are in more than maxOccuranceRate*nrDocuments documents
  *                         are discared. Should be in (0, 1].
  * @param bias             Indicates wheter to include an extra 1 in bag-of-words vectors
  */
class Reader(minOccurrence: Int = 2,
             maxOccurrenceRate: Double = 0.2,
             bias: Boolean = true) {
  private val r = new ReutersRCVStream(new File("./src/main/resources/data/train").getCanonicalPath, ".zip")
  private val wordCounts = scala.collection.mutable.HashMap[String, Int]()
  private val docCount = r.stream.length
  val codes = scala.collection.mutable.Set[String]()


  //count words in files
  for (doc <- r.stream) {
    doc.tokens.distinct.foreach(x => wordCounts(x) = 1 + wordCounts.getOrElse(x, 0))
    doc.codes.foreach(codes += _)
  }

  private val acceptableCount = maxOccurrenceRate * docCount
  val originalDictionarySize = wordCounts.size
  //compute dictionary (remove unusable words)
  val dictionary = wordCounts.filter(x => x._2 < acceptableCount && x._2 >= minOccurrence)
    .keys.toList.sorted.zipWithIndex.toMap

  val reducedDictionarySize = dictionary.size
  private val outLength = reducedDictionarySize + (if (bias) 1 else 0)

  /**
    * Load the datapoints for a given collection.
    * @param collectionName The name of the collection. Either "test", "train" or "validation"
    * @return Stream of datapoints.
    */
  def toBagOfWords(collectionName: String): Stream[DataPoint] =
    toBagOfWords(new ReutersRCVStream(
      new File("./src/main/resources/data/" + collectionName).getCanonicalPath, ".zip").stream)

  /**
    * Load the datapoints for a given stream of documents.
    * @param input The documents.
    * @return Stream of datapoints.
    */
  def toBagOfWords(input: Stream[XMLDocument]): Stream[DataPoint] =
    input.map(doc => {
      val v = new VectorBuilder[Double](outLength)
      doc.tokens.groupBy(identity).mapValues(_.size).toList
        .map { case (key, count) => if (dictionary.contains(key)) (dictionary(key), count) else (-1, 0) }
        .filter(_._1 >= 0).sortBy(_._1)
        .foreach { case (index, count) => v.add(index, count) }
      if (bias) v.add(reducedDictionarySize, 1) //bias
      DataPoint(v.toSparseVector(true, true), doc.codes)
    })

}
