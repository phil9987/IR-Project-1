import java.text.SimpleDateFormat
import java.util.Calendar

/**
  * Created by marc on 13/11/16.
  */

/**
  * Simple logger Class
  * @param name name of the logger
  */
class Logger(name: String) {

  private val positionCounter = scala.collection.mutable.HashMap[String, Int]()
  private val dateFormat = new SimpleDateFormat("HH:mm:ss")

  /**
    * logs the given text
    * @param text Text to be logged.
    */
  def log(text: String): Unit =
  {

    val now = Calendar.getInstance.getTime
    val sb = new StringBuilder
    sb ++= "["
    sb ++= name
    sb ++= "] ["
    sb ++= dateFormat.format(now)
    sb ++= "] "
    sb ++= text
    println(sb.toString)
  }

  /**
    * Reset a log point used in the logPoint function.
    * Sets the counter for the log point to 0.
    * @param logPointName Name of the log point to be reset.
    */
  def restLogPoint(logPointName: String): Unit =
    positionCounter(logPointName) = 0

  /**
    * Similar to log(text).
    * Logs the text only ever logEveryN occurrences to reduce output in busy functions.
    * An internal counter is defined and associated with logPointName.
    * @param text Text to be logged.
    * @param logPointName Name of the logPoint.
    * @param logEveryN Log will be printed every N occurrences.
    */
  def log(text: String, logPointName: String, logEveryN: Int): Unit =
  {
    if (!positionCounter.contains(logPointName)) positionCounter(logPointName) = 0
    if (positionCounter(logPointName) == 0 ) log(text)
    positionCounter.synchronized
    {
      positionCounter(logPointName) = (positionCounter(logPointName) + 1) % logEveryN
    }
  }

}
