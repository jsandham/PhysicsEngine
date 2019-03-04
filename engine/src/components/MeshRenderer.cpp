#include "../../include/components/MeshRenderer.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	meshId = Guid::INVALID;
	materialId = Guid::INVALID;
}

MeshRenderer::MeshRenderer(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	MeshRendererHeader* header = reinterpret_cast<MeshRendererHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
	materialId = header->materialId;
}

MeshRenderer::~MeshRenderer()
{
}