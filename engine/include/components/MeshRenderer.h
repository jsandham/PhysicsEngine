#ifndef MESHRENDERER_H__
#define MESHRENDERER_H__

#include "Component.h"

namespace PhysicsEngine
{
class MeshRenderer : public Component
{
  private:
    Guid mMeshId;
    Guid mMaterialIds[8];

  public:
    int mMaterialCount;
    bool mMeshChanged;
    bool mMaterialChanged;
    bool mIsStatic;
    bool mEnabled;

  public:
    MeshRenderer(World *world, const Id &id);
    MeshRenderer(World *world, const Guid &guid, const Id &id);
    ~MeshRenderer();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void setMesh(const Guid &meshId);
    void setMaterial(const Guid &materialId);
    void setMaterial(const Guid &materialId, int index);

    Guid getMesh() const;
    Guid getMaterial() const;
    Guid getMaterial(int index) const;
    std::vector<Guid> getMaterials() const;
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