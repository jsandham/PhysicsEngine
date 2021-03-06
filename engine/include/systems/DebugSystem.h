#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct DebugSystemHeader
{
    Guid mSystemId;
    int32_t mUpdateOrder;
};
#pragma pack(pop)

class DebugSystem : public System
{
  public:
    DebugSystem();
    DebugSystem(Guid id);
    ~DebugSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &systemId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    void init(World *world);
    void update(const Input &input, const Time &time);
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