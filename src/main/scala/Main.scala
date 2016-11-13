
import ch.ethz.dal.tinyir.processing._

object Main{
  val logger = new Logger("Main")
  def main(args : Array[String]): Unit ={
    logger.log("starting up, got args: " + args.mkString("[", ", ", "]"))
    if (args(0) == "naive"){
      ;
    }else if (args(0) == "logistic"){
/*
      LogisticRegression.train("train")
      LogisticRegression.evaluate("validation")
      LogisticRegression.predict("test")
*/
    }else if (args(0) == "svm"){
      val svm = new SVM(0.1)
      svm.train()
      svm.validate()
      svm.predict()
      //TODO: ask for input path/let prgram use any input path
    }
  }
}
