#include "../../include/components/MeshRenderer.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/GLM.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mMeshGuid = Guid::INVALID;

    for (int i = 0; i < 8; i++)
    {
        mMaterialGuids[i] = Guid::INVALID;
    }

    mMaterialCount = 0;
    mMeshChanged = true;
    mMaterialChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

MeshRenderer::MeshRenderer(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mMeshGuid = Guid::INVALID;

    for (int i = 0; i < 8; i++)
    {
        mMaterialGuids[i] = Guid::INVALID;
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
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["meshId"] = mMeshGuid;
    out["materialCount"] = mMaterialCount;
    for (int i = 0; i < mMaterialCount; i++)
    {
        out["materialIds"].push_back(mMaterialGuids[i]);
    }
    out["isStatic"] = mIsStatic;
    out["enabled"] = mEnabled;
}

void MeshRenderer::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mMeshGuid = YAML::getValue<Guid>(in, "meshId");
    mMaterialCount = YAML::getValue<int>(in, "materialCount");
    for (int i = 0; i < mMaterialCount; i++)
    {
        mMaterialGuids[i] = YAML::getValue<Guid>(in, "materialIds", i);
    }
    mIsStatic = YAML::getValue<bool>(in, "isStatic");
    mEnabled = YAML::getValue<bool>(in, "enabled");

    mMeshChanged = true;
    mMaterialChanged = true;

    mMeshId = mWorld->getIdFromGuid(mMeshGuid);
    for (int i = 0; i < mMaterialCount; i++)
    {
        mMaterialIds[i] = mWorld->getIdFromGuid(mMaterialGuids[i]);
    }
}

int MeshRenderer::getType() const
{
    return PhysicsEngine::MESHRENDERER_TYPE;
}

std::string MeshRenderer::getObjectName() const
{
    return PhysicsEngine::MESHRENDERER_NAME;
}

Guid MeshRenderer::getEntityGuid() const
{
    return mEntityGuid;
}

Guid MeshRenderer::getGuid() const
{
    return mGuid;
}

Id MeshRenderer::getId() const
{
    return mId;
}

void MeshRenderer::setMesh(const Guid &meshGuid)
{
    mMeshId = mWorld->getIdFromGuid(meshGuid);

    mMeshGuid = meshGuid;
    mMeshChanged = true;
}

void MeshRenderer::setMaterial(const Guid &materialGuid)
{
    mMaterialIds[0] = mWorld->getIdFromGuid(materialGuid);

    mMaterialGuids[0] = materialGuid;
    mMaterialChanged = true;
    if (mMaterialCount == 0)
    {
        mMaterialCount = 1;
    }
}

void MeshRenderer::setMaterial(const Guid &materialGuid, int index)
{
    if (index >= 0 && index < 8)
    {
        mMaterialIds[index] = mWorld->getIdFromGuid(materialGuid);

        mMaterialGuids[index] = materialGuid;
        mMaterialChanged = true;
    }
}

Guid MeshRenderer::getMesh() const
{
    return mMeshGuid;
}

Guid MeshRenderer::getMaterial() const
{
    return mMaterialGuids[0];
}

Guid MeshRenderer::getMaterial(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mMaterialGuids[index];
    }

    return Guid::INVALID;
}

std::vector<Guid> MeshRenderer::getMaterials() const
{
    std::vector<Guid> materials;
    for (int i = 0; i < 8; i++)
    {
        if (mMaterialGuids[i] != Guid::INVALID)
        {
            materials.push_back(mMaterialGuids[i]);
        }
    }

    return materials;
}

Id MeshRenderer::getMeshId() const
{
    return mMeshId;
}

Id MeshRenderer::getMaterialId() const
{
    return mMaterialIds[0];
}

Id MeshRenderer::getMaterialId(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mMaterialIds[index];
    }

    return Id::INVALID;
}