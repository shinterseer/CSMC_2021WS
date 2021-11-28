#include <sys/time.h>


/** @brief Simple timer class based on gettimeofday (POSIX).
  *
  */
class Timer
{
public:

  Timer() : ts(0)
  {}

  void reset()
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
  }

  double get() const
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

    return static_cast<double>(end_time-ts) / 1000000.0;
  }

private:
  double ts;
};
