#include "../../include/components/MeshRenderer.h"

#include "../../include/core/GLM.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer(World* world) : Component(world)
{
    mMeshId = -1;

    for (int i = 0; i < 8; i++)
    {
        mMaterialIds[i] = -1;
    }

    mMaterialCount = 0;
    mMeshChanged = true;
    mMaterialChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

MeshRenderer::MeshRenderer(World* world, Id id) : Component(world, id)
{
    mMeshId = -1;

    for (int i = 0; i < 8; i++)
    {
        mMaterialIds[i] = -1;
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

    out["meshId"] = mWorld->getGuidOf(mMeshId);
    out["materialCount"] = mMaterialCount;
    for (int i = 0; i < mMaterialCount; i++)
    {
        out["materialIds"].push_back(mWorld->getGuidOf(mMaterialIds[i]));
    }
    out["isStatic"] = mIsStatic;
    out["enabled"] = mEnabled;
}

void MeshRenderer::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mMeshId = mWorld->getIdOf(YAML::getValue<Guid>(in, "meshId"));
    mMaterialCount = YAML::getValue<int>(in, "materialCount");
    for (int i = 0; i < mMaterialCount; i++)
    {
        mMaterialIds[i] = mWorld->getIdOf(YAML::getValue<Guid>(in, "materialIds", i));
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

void MeshRenderer::setMesh(Id meshId)
{
    mMeshId = meshId;
    mMeshChanged = true;
}

void MeshRenderer::setMaterial(Id materialId)
{
    mMaterialIds[0] = materialId;
    mMaterialChanged = true;
    if (mMaterialCount == 0)
    {
        mMaterialCount = 1;
    }
}

void MeshRenderer::setMaterial(Id materialId, int index)
{
    if (index >= 0 && index < 8)
    {
        mMaterialIds[index] = materialId;
        mMaterialChanged = true;
    }
}

Id MeshRenderer::getMesh() const
{
    return mMeshId;
}

Id MeshRenderer::getMaterial() const
{
    return mMaterialIds[0];
}

Id MeshRenderer::getMaterial(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mMaterialIds[index];
    }

    return -1;
}

std::vector<Id> MeshRenderer::getMaterials() const
{
    std::vector<Id> materials;
    for (int i = 0; i < 8; i++)
    {
        if (mMaterialIds[i] != -1)
        {
            materials.push_back(mMaterialIds[i]);
        }
    }

    return materials;
}