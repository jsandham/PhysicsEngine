#ifndef TIME_H__
#define TIME_H__

#include <stdint.h>

namespace PhysicsEngine
{
struct Time
{
    int64_t mFrameCount;          // frame count since application started
    double mStartTime;            // start time at beginning of frame (seconds)
    double mEndTime;              // end time at end of frame (seconds)
    double mDeltaTime;            // elapsed time of frame (seconds)
    double mDeltaTimeHistory[64]; // history of elapsed time of frame (seconds)
};

float getFPS(const Time &time);
float getSmoothedFPS(const Time &time);
double getSmoothedDeltaTime(const Time &time);

Time &getTime();
} // namespace PhysicsEngine

#endif