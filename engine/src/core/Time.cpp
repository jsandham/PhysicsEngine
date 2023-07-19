#include "../../include/core/Time.h"

using namespace PhysicsEngine;

Time global_time = {};

Time &PhysicsEngine::getTime()
{
    return global_time;
}

float PhysicsEngine::getFPS(const Time &time)
{
    return static_cast<float>(1 / time.mDeltaTime);
}

float PhysicsEngine::getSmoothedFPS(const Time &time)
{
    return static_cast<float>(1 / getSmoothedDeltaTime(time));
}

double PhysicsEngine::getSmoothedDeltaTime(const Time &time)
{
    double temp = 0;
    for (int i = 0; i < 64; i++)
    {
        temp += time.mDeltaTimeHistory[i];
    }

    return (temp / 64);
}