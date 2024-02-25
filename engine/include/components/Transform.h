#ifndef TRANSFORM_H__
#define TRANSFORM_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentYaml.h"
#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;
class Entity;

struct TransformData
{
    glm::vec3 mPosition;
    glm::quat mRotation;
    glm::vec3 mScale;

    TransformData();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    glm::mat4 getModelMatrix() const;
    glm::vec3 getForward() const;
    glm::vec3 getUp() const;
    glm::vec3 getRight() const;
};


//class TransformTag
//{
//  private:
//    Guid mGuid;
//    Id mId;
//    Guid mEntityGuid;
//
//    World *mWorld;
//
//  public:
//    HideFlag mHide;
//    bool mEnabled;
//};


class Transform
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
     HideFlag mHide;
     bool mEnabled;

  public:
    Transform(World *world, const Id &id);
    Transform(World *world, const Guid &guid, const Id &id);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

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

    Entity *getEntity() const;

    template <typename T>
    T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

    static bool decompose(const glm::mat4 &model, glm::vec3 &translation, glm::quat &rotation, glm::vec3 &scale);

  private:
    static void v3Scale(glm::vec3 &v, float desiredLength);

    friend class Scene;
};

} // namespace PhysicsEngine

#endif