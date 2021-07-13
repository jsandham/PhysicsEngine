#include "../../include/components/MeshRenderer.h"

#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer() : Component()
{
    mMeshId = Guid::INVALID;

    for (int i = 0; i < 8; i++)
    {
        mMaterialIds[i] = Guid::INVALID;
    }

    mMaterialCount = 0;
    mMeshChanged = true;
    mMaterialChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

MeshRenderer::MeshRenderer(Guid id) : Component(id)
{
    mMeshId = Guid::INVALID;

    for (int i = 0; i < 8; i++)
    {
        mMaterialIds[i] = Guid::INVALID;
    }

    mMaterialCount = 0;
    mMeshChanged = true;
    mMaterialChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

MeshRenderer::~MeshRenderer()
{
}

void MeshRenderer::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["meshId"] = mMeshId;
    out["materialCount"] = mMaterialCount;
    for (int i = 0; i < mMaterialCount; i++)
    {
        out["materialIds"].push_back(mMaterialIds[i]);
    }
    out["isStatic"] = mIsStatic;
    out["enabled"] = mEnabled;
}

void MeshRenderer::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mMeshId = YAML::getValue<Guid>(in, "meshId");
    mMaterialCount = YAML::getValue<int>(in, "materialCount");
    for (int i = 0; i < mMaterialCount; i++)
    {
        mMaterialIds[i] = YAML::getValue<Guid>(in, "materialIds", i);
    }
    mIsStatic = YAML::getValue<bool>(in, "isStatic");
    mEnabled = YAML::getValue<bool>(in, "enabled");

    mMeshChanged = true;
    mMaterialChanged = true;
}

int MeshRenderer::getType() const
{
    return PhysicsEngine::MESHRENDERER_TYPE;
}

std::string MeshRenderer::getObjectName() const
{
    return PhysicsEngine::MESHRENDERER_NAME;
}

void MeshRenderer::setMesh(Guid meshId)
{
    mMeshId = meshId;
    mMeshChanged = true;
}

void MeshRenderer::setMaterial(Guid materialId)
{
    mMaterialIds[0] = materialId;
    mMaterialChanged = true;
    if (mMaterialCount == 0)
    {
        mMaterialCount = 1;
    }
}

void MeshRenderer::setMaterial(Guid materialId, int index)
{
    if (index >= 0 && index < 8)
    {
        mMaterialIds[index] = materialId;
        mMaterialChanged = true;
    }
}

Guid MeshRenderer::getMesh() const
{
    return mMeshId;
}

Guid MeshRenderer::getMaterial() const
{
    return mMaterialIds[0];
}

Guid MeshRenderer::getMaterial(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mMaterialIds[index];
    }

    return Guid::INVALID;
}

std::vector<Guid> MeshRenderer::getMaterials() const
{
    std::vector<Guid> materials;
    for (int i = 0; i < 8; i++)
    {
        if (mMaterialIds[i] != Guid::INVALID)
        {
            materials.push_back(mMaterialIds[i]);
        }
    }

    return materials;
}