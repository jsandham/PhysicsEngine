#ifndef CLEANUPSYSTEM_H__
#define CLEANUPSYSTEM_H__

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

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;
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