#ifndef TIME_H__
#define TIME_H__

namespace PhysicsEngine
{
struct Time
{
    size_t mFrameCount; // frame count since application started
    double mStartTime; // start time at beginning of frame (seconds)
    double mEndTime; // end time at end of frame (seconds)
    double mDeltaTime; // elapsed time of frame (seconds)
};

float getFPS(const Time& time);
} // namespace PhysicsEngine

#endif