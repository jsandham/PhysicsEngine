#ifndef TRANSFORM_H__
#define TRANSFORM_H__

#include <vector>

#define GLM_FORCE_RADIANS

#include "Component.h"

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/epsilon.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"

#include "glm/geometric.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

namespace PhysicsEngine
{
class Transform : public Component
{
  public:
    Guid mParentId;
    glm::vec3 mPosition;
    glm::quat mRotation;
    glm::vec3 mScale;

  public:
    Transform(World *world);
    Transform(World *world, const Guid& id);
    ~Transform();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

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