#include "../../include/components/MeshRenderer.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	meshId = Guid::INVALID;

	for(int i = 0; i < 8; i++){
		materialIds[i] = Guid::INVALID;
	}

	materialCount = 0;
	isStatic = true;
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
	return serialize(componentId, entityId);
}

std::vector<char> MeshRenderer::serialize(Guid componentId, Guid entityId) const
{
	MeshRendererHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.meshId = meshId;
	for (int i = 0; i < 8; i++) {
		header.materialIds[i] = materialIds[i];
	}
	header.materialCount = materialCount;
	header.isStatic = isStatic;

	std::vector<char> data(sizeof(MeshRendererHeader));

	memcpy(&data[0], &header, sizeof(MeshRendererHeader));

	return data;
}

void MeshRenderer::deserialize(std::vector<char> data)
{
	MeshRendererHeader* header = reinterpret_cast<MeshRendererHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
	for(int i = 0; i < 8; i++){
		materialIds[i] = header->materialIds[i];
	}
	materialCount = header->materialCount;
	isStatic = header->isStatic;
}