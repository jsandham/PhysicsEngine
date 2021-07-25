#ifndef RIGIDBODY_H__
#define RIGIDBODY_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
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
    Rigidbody(World* world);
    Rigidbody(World* world, Guid id);
    ~Rigidbody();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;
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