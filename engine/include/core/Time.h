#ifndef TIME_H__
#define TIME_H__

namespace PhysicsEngine
{
struct Time
{
    size_t frameCount;
    float time;
    float deltaTime;
    size_t deltaCycles;
};

float getFPS(Time time);
} // namespace PhysicsEngine

#endif