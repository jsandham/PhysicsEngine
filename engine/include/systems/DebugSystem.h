#ifndef DEBUGSYSTEM_H__
#define DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
class DebugSystem : public System
{
  public:
    DebugSystem();
    DebugSystem(Guid id);
    ~DebugSystem();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;
};

template <> struct SystemType<DebugSystem>
{
    static constexpr int type = PhysicsEngine::DEBUGSYSTEM_TYPE;
};
template <> struct IsSystemInternal<DebugSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif