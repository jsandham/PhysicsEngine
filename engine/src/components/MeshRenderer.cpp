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
   
    mMeshChanged = true;
    mMaterialChanged = true;
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