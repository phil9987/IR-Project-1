import scala.collection.mutable.PriorityQueue

class cutoffFinder(proportion : Double){

  val prop = proportion
  object MinOrder extends Ordering[Double]{
    def compare(x:Double, y:Double) = y compare x
  }

  var maxHeap = new PriorityQueue[Double]()
  var minHeap = PriorityQueue.empty(MinOrder)

  var totalLength = 0.0
  var maxHeapLength = 0.0
  var minHeapLength = 0.0

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

  def getCutoff(): Double ={
    //println(s" number of elements in each heap : maxHeap : $maxHeapLength  |    minHeap : $minHeapLength")
    //println(s"total elements $totalLength")
    (maxHeap.head + minHeap.head) / 2.0
  }
}