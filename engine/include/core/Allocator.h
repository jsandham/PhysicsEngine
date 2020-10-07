#ifndef __ALLOCATOR_H__
#define __ALLOCATOR_H__

namespace PhysicsEngine
{
class Allocator
{
  public:
    Allocator();
    virtual ~Allocator() = 0;

    virtual size_t getCount() const = 0;
};
} // namespace PhysicsEngine

#endif
