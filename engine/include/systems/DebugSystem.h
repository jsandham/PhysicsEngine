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
  private:
    Material *mColorMat;
    Shader *mColorShader;

  public:
    DebugSystem();
    DebugSystem(const std::vector<char> &data);
    ~DebugSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid systemId) const;
    void deserialize(const std::vector<char> &data);

    void init(World *world);
    void update(Input input, Time time);
};

template <typename T> struct IsDebugSystem
{
    static constexpr bool value = false;
};

template <> struct SystemType<DebugSystem>
{
    static constexpr int type = PhysicsEngine::DEBUGSYSTEM_TYPE;
};
template <> struct IsDebugSystem<DebugSystem>
{
    static constexpr bool value = true;
};
template <> struct IsSystem<DebugSystem>
{
    static constexpr bool value = true;
};
template <> struct IsSystemInternal<DebugSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif