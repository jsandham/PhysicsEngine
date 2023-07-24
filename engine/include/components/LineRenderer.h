#ifndef LINERENDERER_H__
#define LINERENDERER_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

class LineRenderer
{
  public:
    HideFlag mHide;

  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

    glm::vec3 mStart;
    glm::vec3 mEnd;
    bool mEnabled;

    Guid mMaterialId;

  public:
    LineRenderer(World *world, const Id &id);
    LineRenderer(World *world, const Guid &guid, const Id &id);
    ~LineRenderer();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;
  
  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif