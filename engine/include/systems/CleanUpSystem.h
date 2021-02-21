#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

namespace PhysicsEngine
{
class CleanUpSystem : public System
{
  public:
    CleanUpSystem();
    CleanUpSystem(Guid id);
    ~CleanUpSystem();

    virtual void serialize(std::ostream& out) const;
    virtual void deserialize(std::istream& in);

    void init(World *world);
    void update(const Input &input, const Time &time);
};

template <> struct SystemType<CleanUpSystem>
{
    static constexpr int type = PhysicsEngine::CLEANUPSYSTEM_TYPE;
};
template <> struct IsSystemInternal<CleanUpSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif