#include "../../include/components/MeshRenderer.h"

#include "../../include/core/Serialize.h"

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

std::vector<char> MeshRenderer::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> MeshRenderer::serialize(const Guid &componentId, const Guid &entityId) const
{
    MeshRendererHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mMeshId = mMeshId;
    for (int i = 0; i < 8; i++)
    {
        header.mMaterialIds[i] = mMaterialIds[i];
    }
    header.mMaterialCount = static_cast<int32_t>(mMaterialCount);
    header.mIsStatic = static_cast<uint8_t>(mIsStatic);
    header.mEnabled = static_cast<uint8_t>(mEnabled);

    std::vector<char> data(sizeof(MeshRendererHeader));

    memcpy(&data[0], &header, sizeof(MeshRendererHeader));

    return data;
}

void MeshRenderer::deserialize(const std::vector<char> &data)
{
    const MeshRendererHeader *header = reinterpret_cast<const MeshRendererHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mMeshId = header->mMeshId;
    for (int i = 0; i < 8; i++)
    {
        mMaterialIds[i] = header->mMaterialIds[i];
    }
    mMaterialCount = static_cast<int>(header->mMaterialCount);
    mIsStatic = static_cast<bool>(header->mIsStatic);
    mEnabled = static_cast<bool>(header->mEnabled);
    mMeshChanged = true;
    mMaterialChanged = true;
}

void MeshRenderer::serialize(std::ostream& out) const
{
    Component::serialize(out);

    PhysicsEngine::write<Guid>(out, mMeshId);
    for (int i = 0; i < 8; i++)
    {
        PhysicsEngine::write<Guid>(out, mMaterialIds[i]);
    }
    PhysicsEngine::write<int>(out, mMaterialCount);
    PhysicsEngine::write<bool>(out, mIsStatic);
    PhysicsEngine::write<bool>(out, mEnabled);
    PhysicsEngine::write<bool>(out, mMeshChanged);
    PhysicsEngine::write<bool>(out, mMaterialChanged);
}

void MeshRenderer::deserialize(std::istream& in)
{
    Component::deserialize(in);

    PhysicsEngine::read<Guid>(in, mMeshId);
    for (int i = 0; i < 8; i++)
    {
        PhysicsEngine::read<Guid>(in, mMaterialIds[i]);
    }
    PhysicsEngine::read<int>(in, mMaterialCount);
    PhysicsEngine::read<bool>(in, mIsStatic);
    PhysicsEngine::read<bool>(in, mEnabled);
    PhysicsEngine::read<bool>(in, mMeshChanged);
    PhysicsEngine::read<bool>(in, mMaterialChanged);
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