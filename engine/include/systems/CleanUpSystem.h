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
    CleanUpSystem(World *world, const Id &id);
    CleanUpSystem(World *world, const Guid &guid, const Id &id);
    ~CleanUpSystem();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

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