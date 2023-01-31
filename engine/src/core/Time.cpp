#include "../../include/core/Time.h"

using namespace PhysicsEngine;

Time global_time = {};

Time &PhysicsEngine::getTime()
{
    return global_time;
}

float PhysicsEngine::getFPS(const Time& time)
{
    return static_cast<float>(1 / time.mDeltaTime);
}