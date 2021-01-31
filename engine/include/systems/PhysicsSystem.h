#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../components/Collider.h"
#include "../components/Rigidbody.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct PhysicsSystemHeader
{
    Guid mSystemId;
    int32_t mUpdateOrder;
    float mTimestep;
    float mGravity;
};
#pragma pack(pop)

class PhysicsSystem : public System
{
  private:
    std::vector<Collider *> mColliders;
    std::vector<Rigidbody *> mRigidbodies;

    float mTimestep;
    float mGravity;

  public:
    PhysicsSystem();
    PhysicsSystem(Guid id);
    ~PhysicsSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &systemId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    void init(World *world);
    void update(const Input &input, const Time &time);
};

template <> struct SystemType<PhysicsSystem>
{
    static constexpr int type = PhysicsEngine::PHYSICSSYSTEM_TYPE;
};
template <> struct IsSystemInternal<PhysicsSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif