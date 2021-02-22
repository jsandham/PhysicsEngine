#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtx/quaternion.hpp"

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
    Transform();
    Transform(Guid id);
    ~Transform();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

    glm::mat4 getModelMatrix() const;
    glm::vec3 getForward() const;
    glm::vec3 getUp() const;
    glm::vec3 getRight() const;
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