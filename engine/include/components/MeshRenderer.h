#ifndef MESHRENDERER_H__
#define MESHRENDERER_H__

#include "Component.h"

namespace PhysicsEngine
{
class MeshRenderer : public Component
{
  private:
    Id mMeshId;
    Id mMaterialIds[8];

  public:
    int mMaterialCount;
    bool mMeshChanged;
    bool mMaterialChanged;
    bool mIsStatic;
    bool mEnabled;

  public:
    MeshRenderer(World *world);
    MeshRenderer(World *world, Id id);
    ~MeshRenderer();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void setMesh(Id meshId);
    void setMaterial(Id materialId);
    void setMaterial(Id materialId, int index);

    Id getMesh() const;
    Id getMaterial() const;
    Id getMaterial(int index) const;
    std::vector<Id> getMaterials() const;
};

template <> struct ComponentType<MeshRenderer>
{
    static constexpr int type = PhysicsEngine::MESHRENDERER_TYPE;
};

template <> struct IsComponentInternal<MeshRenderer>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif