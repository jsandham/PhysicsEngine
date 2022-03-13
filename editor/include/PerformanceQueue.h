#ifndef PERFORMANCE_QUEUE_H__
#define PERFORMANCE_QUEUE_H__

#include <vector>

namespace PhysicsEditor
{
class PerformanceQueue
{
  private:
    int mNumberOfSamples;
    int mIndex;
    std::vector<float> mQueue;
    std::vector<float> mData;

  public:
    PerformanceQueue();
    ~PerformanceQueue();

    void setNumberOfSamples(int numberOfSamples);
    void addSample(float value);
    void clear();
    std::vector<float> getData();
};
} // namespace PhysicsEditor

#endif
