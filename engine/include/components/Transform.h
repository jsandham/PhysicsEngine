#ifndef TRANSFORM_H__
#define TRANSFORM_H__

#define GLM_FORCE_RADIANS

#include "Component.h"

#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"

namespace PhysicsEngine
{
class Transform : public Component
{
  private:
    glm::vec3 mPosition;
    glm::quat mRotation;
    glm::vec3 mScale;

  public:
    Transform(World *world, const Id& id);
    Transform(World *world, const Guid& guid, const Id& id);
    ~Transform();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void setPosition(const glm::vec3 &position);
    void setRotation(const glm::quat &rotation);
    void setScale(const glm::vec3 &scale);

    glm::vec3 getPosition() const;
    glm::quat getRotation() const;
    glm::vec3 getScale() const;

    glm::mat4 getModelMatrix() const;
    glm::vec3 getForward() const;
    glm::vec3 getUp() const;
    glm::vec3 getRight() const;

    static bool decompose(const glm::mat4 &model, glm::vec3 &translation, glm::quat &rotation, glm::vec3 &scale);

  private:
    static void v3Scale(glm::vec3 &v, float desiredLength);
};

template <> struct ComponentType<Transform>
{
    static constexpr int type = PhysicsEngine::TRANSFORM_TYPE;
};

template <> struct IsComponentInternal<Transform>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif