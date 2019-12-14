#include "../../include/components/MeshRenderer.h"

#include "../../include/core/PoolAllocator.h"

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

std::vector<char> MeshRenderer::serialize()
{
	MeshRendererHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.meshId = meshId;
	for(int i = 0; i < 8; i++){
		header.materialIds[i] = materialIds[i];
	}
	header.materialCount = materialCount;
	header.isStatic = isStatic;

	int numberOfBytes = sizeof(MeshRendererHeader);

	std::vector<char> data(numberOfBytes);

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