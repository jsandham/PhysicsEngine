#include "../../include/core/Time.h"

using namespace PhysicsEngine;

float PhysicsEngine::getFPS(const Time& time)
{
    return static_cast<float>(1 / time.mDeltaTime);
}