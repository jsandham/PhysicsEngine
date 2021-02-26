#ifndef ALLOCATOR_H__
#define ALLOCATOR_H__

namespace PhysicsEngine
{
class Allocator
{
  public:
    Allocator();
    virtual ~Allocator() = 0;

    virtual size_t getCount() const = 0;
    virtual size_t getCapacity() const = 0;
};
} // namespace PhysicsEngine

#endif
