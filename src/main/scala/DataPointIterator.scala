import java.io.File
import java.io.FileNotFoundException
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import scala.collection.JavaConversions.enumerationAsScalaIterator
import scala.util.Success
import scala.util.Try
import ch.ethz.dal.tinyir.processing._
import breeze.linalg.{SparseVector, Vector, VectorBuilder}

class DataPointIterator(corpusType : String, linkedReader : Reader, bias : Boolean = true) extends Iterator[DataPoint] {

  val resourceFolder = getClass.getResource("/data/" + corpusType + "/").getPath
  println(s"loaded resource folder : $resourceFolder")
  // we will use this to iterate over ZIP entries
  var xmlIterator :  Iterator[ZipEntry] = _
  var currentZip : ZipFile = _

  val dirFile = new File(resourceFolder)
    if(dirFile == null || !dirFile.isDirectory())
      throw new FileNotFoundException("No suche directory: " + dirFile)
    else
      Try(new ZipFile(resourceFolder + "/" + corpusType + ".zip"))  match {
        case Success(zip) => { currentZip = zip; xmlIterator =  zip.entries.toIterator }
        case _ => Iterator.empty
      }


  def hasNext : Boolean = {
    // We still have xmls left?
    if(xmlIterator.hasNext)
      return true
    false
  }

  def next() : DataPoint = {
    if(!hasNext)
      throw new NoSuchElementException()

    // Open file, parse it and close it
    val entry = xmlIterator.next
    val is = currentZip.getInputStream(entry)
    val xml = new ReutersRCVParse(is)
    is.close()

    val v = new VectorBuilder[Double](linkedReader.outLength)
    xml.tokens.groupBy(identity).mapValues(_.size).toList
      .map { case (key, count) => if (linkedReader.dictionary.contains(key)) (linkedReader.dictionary(key), count) else (-1, 0) }
      .filter(_._1 >= 0).sortBy(_._1)
      .foreach { case (index, count) => v.add(index, count) }
    if (bias) v.add(linkedReader.reducedDictionarySize, 1) //bias
    DataPoint(v.toSparseVector(true, true), xml.codes.intersect(possibleCodes.topicCodes))
  }
}
