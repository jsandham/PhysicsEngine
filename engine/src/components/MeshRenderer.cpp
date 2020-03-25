#include "../../include/components/MeshRenderer.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	mMeshId = Guid::INVALID;

	for(int i = 0; i < 8; i++){
		mMaterialIds[i] = Guid::INVALID;
	}

	mMaterialCount = 0;
	mIsStatic = true;
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
}