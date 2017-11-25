//
// Created by spades on 25/05/17.
//

#ifndef CGMAPPING_TIMER_H
#define CGMAPPING_TIMER_H

#include <iostream>
#include <chrono>

namespace cgmapping {

class Timer {

 private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point <clock_> beg_;

 public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed_in_sec() const {
      return std::chrono::duration_cast<second_>
        (clock_::now() - beg_).count();
  }
  double elapsed_in_msec() const {
      return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count()*1000;
  }
  double elapsed_in_nsec() const {
      return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count()*1000000;
  }

};
}
#endif //CGMAPPING_TIMER_H
