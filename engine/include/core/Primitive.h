#ifndef __PRIMITIVE_H__
#define __PRIMITIVE_H__

namespace PhysicsEngine
{
typedef enum PrimitiveType
{
    Sphere,
    Bounds,
    Capsule
};

class Primitive
{
    virtual PrimitiveType getPrimitiveType() = 0;
};
} // namespace PhysicsEngine

#endif