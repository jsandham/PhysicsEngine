#ifndef MESHRENDERER_H__
#define MESHRENDERER_H__

#include <vector>

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
    MeshRenderer();
    MeshRenderer(Guid id);
    ~MeshRenderer();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;

    void setMesh(Guid meshId);
    void setMaterial(Guid materialId);
    void setMaterial(Guid materialId, int index);

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