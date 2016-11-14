
import ch.ethz.dal.tinyir.processing._

object Main{
  val logger = new Logger("Main")
  def main(args : Array[String]): Unit ={
    //TODO create menu
    logger.log("starting up, got args: " + args.mkString("[", ", ", "]"))
    if (args(0) == "bayes"){
      NaiveBayes.train()
      NaiveBayes.validate()
      NaiveBayes.predict()

    }else if (args(0) == "logistic"){
/*
      LogisticRegression.train("train")
      LogisticRegression.evaluate("validation")
      LogisticRegression.predict("test")
*/
    }else if (args(0) == "svm"){
      val svm = new SVM(1e-4) //TODO
      svm.train()
      svm.validate()
      svm.predict("ir-2016-1-project-7-svm.txt")
    }
  }
}
