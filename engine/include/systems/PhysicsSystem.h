#ifndef PHYSICSSYSTEM_H__
#define PHYSICSSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../components/Collider.h"
#include "../components/Rigidbody.h"

namespace PhysicsEngine
{
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

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;
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