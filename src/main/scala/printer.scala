object printer{
  var verbose = true
  var count = 0
  def print(str : String, sieve : Int = 1) {
    if (verbose) {
      count = ( count + 1 ) % sieve

      if (count == 0) {
        println(str)
      }
    }
  }
}