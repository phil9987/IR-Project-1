import java.io.File
import java.io.FileNotFoundException
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import scala.collection.JavaConversions.enumerationAsScalaIterator
import scala.collection.mutable.Queue
import scala.util.Success
import scala.util.Try
import ch.ethz.dal.tinyir.processing._


class ReutersCorpusIterator(corpusType : String) extends Iterator[ReutersRCVParse] {

  val resourceFolder = getClass.getResource("/data/").getPath
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

  def next() : ReutersRCVParse = {
    if(!hasNext)
      throw new NoSuchElementException()

    // Open file, parse it and close it
    val entry = xmlIterator.next
    val is = currentZip.getInputStream(entry)
    val xml = new ReutersRCVParse(is)
    is.close()
    xml
  }
}
