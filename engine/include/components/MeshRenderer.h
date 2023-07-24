#ifndef MESHRENDERER_H__
#define MESHRENDERER_H__

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

class MeshRenderer
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

    Guid mMeshId;
    Guid mMaterialIds[8];

  public:
    HideFlag mHide;

    int mMaterialCount;
    bool mMeshChanged;
    bool mMaterialChanged;
    bool mIsStatic;
    bool mEnabled;

  public:
    MeshRenderer(World *world, const Id &id);
    MeshRenderer(World *world, const Guid &guid, const Id &id);
    ~MeshRenderer();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    void setMesh(const Guid &meshId);
    void setMaterial(const Guid &materialId);
    void setMaterial(const Guid &materialId, int index);

    Guid getMesh() const;
    Guid getMaterial() const;
    Guid getMaterial(int index) const;
    std::vector<Guid> getMaterials() const;

  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif