#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct RigidbodyHeader
{
    Guid mComponentId;
    Guid mEntityId;
    glm::mat3 mInertiaTensor;
    glm::vec3 mVelocity;
    glm::vec3 mAngularVelocity;
    glm::vec3 mCentreOfMass;
    float mMass;
    float mDrag;
    float mAngularDrag;
    uint8_t mUseGravity;
};
#pragma pack(pop)

class Rigidbody : public Component
{
  public:
    bool mUseGravity;
    float mMass;
    float mDrag;
    float mAngularDrag;

    glm::vec3 mVelocity;
    glm::vec3 mAngularVelocity;
    glm::vec3 mCentreOfMass;
    glm::mat3 mInertiaTensor;

    // leap-frog
    glm::vec3 mHalfVelocity;

  public:
    Rigidbody();
    Rigidbody(Guid id);
    ~Rigidbody();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);
};

template <> struct ComponentType<Rigidbody>
{
    static constexpr int type = PhysicsEngine::RIGIDBODY_TYPE;
};

template <> struct IsComponentInternal<Rigidbody>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif