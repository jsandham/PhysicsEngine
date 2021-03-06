#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct CleanUpSystemHeader
{
    Guid mSystemId;
    int32_t mUpdateOrder;
};
#pragma pack(pop)

class CleanUpSystem : public System
{
  public:
    CleanUpSystem();
    CleanUpSystem(Guid id);
    ~CleanUpSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &systemId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

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