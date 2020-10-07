#include "../../include/core/Time.h"

using namespace PhysicsEngine;

float PhysicsEngine::getFPS(Time time)
{
    return 1000 / time.deltaTime;
}