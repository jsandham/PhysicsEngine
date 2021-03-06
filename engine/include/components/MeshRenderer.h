#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include <vector>

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct MeshRendererHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Guid mMeshId;
    Guid mMaterialIds[8];
    int32_t mMaterialCount;
    uint8_t mIsStatic;
    uint8_t mEnabled;
};
#pragma pack(pop)

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

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

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