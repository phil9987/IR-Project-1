import scala.collection.mutable.PriorityQueue

/**
  * Helper class to find cutoff values.
  * @param proportion : the proportion of probabilities that have to be above the cutoff
  */
class cutoffFinder(proportion : Double){

  val prop = proportion
  object MinOrder extends Ordering[Double]{
    def compare(x:Double, y:Double) = y compare x
  }

  private var maxHeap = new PriorityQueue[Double]()
  private var minHeap = PriorityQueue.empty(MinOrder)

  var totalLength = 0.0
  var maxHeapLength = 0.0
  var minHeapLength = 0.0

  /**
    * adds the Probability to the list of stored probabilities
    * @param newDouble : The new probability to add
    */
  def add(newDouble : Double): Unit ={
    totalLength += 1
    if (maxHeapLength == 0 && minHeapLength == 0) {
      maxHeap.enqueue(newDouble)
      maxHeapLength += 1.0
    }
    else if (newDouble < maxHeap.head){
      maxHeap.enqueue(newDouble)
      maxHeapLength += 1.0
    }
    else{
      minHeap.enqueue(newDouble)
      minHeapLength += 1.0
    }

    //rebalance
    while (maxHeapLength / totalLength > 1 - prop){
      minHeap.enqueue(maxHeap.dequeue)
      minHeapLength += 1
      maxHeapLength -= 1
    }
    while (minHeapLength / totalLength > prop) {
      maxHeap.enqueue(minHeap.dequeue)
      minHeapLength -= 1
      maxHeapLength += 1
    }

  }

  /**
    * Returns the obtained cutoff value
    * @return : the cutoff value (between the two heaps)
    */
  def getCutoff(): Double ={
    (maxHeap.head + minHeap.head) / 2.0
  }
}