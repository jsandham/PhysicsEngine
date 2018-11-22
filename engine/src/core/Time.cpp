#include "../../include/core/Time.h"

using namespace PhysicsEngine;

int Time::frameCount = 0;
int Time::deltaCycles = 0;
int Time::gpuDeltaCycles = 0;
float Time::time = 0.0f;
float Time::deltaTime = 0.0f;
float Time::gpuDeltaTime = 0.0f;