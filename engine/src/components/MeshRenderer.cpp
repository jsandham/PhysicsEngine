#include "../../include/components/MeshRenderer.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	mMeshId = Guid::INVALID;

	for(int i = 0; i < 8; i++){
		mMaterialIds[i] = Guid::INVALID;
	}

	mMaterialCount = 0;
	mMeshChanged = true;
	mMaterialChanged = true;
	mIsStatic = true;
	mEnabled = true;
}

MeshRenderer::MeshRenderer(std::vector<char> data)
{
	deserialize(data);
}

MeshRenderer::~MeshRenderer()
{
}

std::vector<char> MeshRenderer::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> MeshRenderer::serialize(Guid componentId, Guid entityId) const
{
	MeshRendererHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mMeshId = mMeshId;
	for (int i = 0; i < 8; i++) {
		header.mMaterialIds[i] = mMaterialIds[i];
	}
	header.mMaterialCount = mMaterialCount;
	header.mIsStatic = mIsStatic;
	header.mEnabled = mEnabled;

	std::vector<char> data(sizeof(MeshRendererHeader));

	memcpy(&data[0], &header, sizeof(MeshRendererHeader));

	return data;
}

void MeshRenderer::deserialize(std::vector<char> data)
{
	MeshRendererHeader* header = reinterpret_cast<MeshRendererHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mMeshId = header->mMeshId;
	for(int i = 0; i < 8; i++){
		mMaterialIds[i] = header->mMaterialIds[i];
	}
	mMaterialCount = header->mMaterialCount;
	mIsStatic = header->mIsStatic;
	mEnabled = header->mEnabled;
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
}

void MeshRenderer::setMaterial(Guid materialId, int index)
{
	if (index >= 0 && index < 8) {
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
	if (index >= 0 && index < 8) {
		return mMaterialIds[index];
	}

	return Guid::INVALID;
}

std::vector<Guid> MeshRenderer::getMaterials() const
{
	std::vector<Guid> materials;
	for (int i = 0; i < 8; i++) {
		if (mMaterialIds[i] != Guid::INVALID) {
			materials.push_back(mMaterialIds[i]);
		}
	}

	return materials;
}