#include "../../include/components/MeshRenderer.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	meshId = Guid::INVALID;

	for(int i = 0; i < 8; i++){
		materialIds[i] = Guid::INVALID;
	}

	isStatic = true;
}

MeshRenderer::MeshRenderer(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	MeshRendererHeader* header = reinterpret_cast<MeshRendererHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
	for(int i = 0; i < 8; i++){
		materialIds[i] = header->materialIds[i];
	}
	isStatic = header->isStatic;
}

MeshRenderer::~MeshRenderer()
{
}