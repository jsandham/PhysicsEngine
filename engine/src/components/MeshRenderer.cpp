#include "../../include/components/MeshRenderer.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	meshId = Guid::INVALID;
	materialId = Guid::INVALID;
}

MeshRenderer::MeshRenderer(unsigned char* data)
{
	
}

MeshRenderer::~MeshRenderer()
{
}

void* MeshRenderer::operator new(size_t size)
{
	return getAllocator<MeshRenderer>().allocate();
}

void MeshRenderer::operator delete(void*)
{

}

void MeshRenderer::load(MeshRendererData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	meshId = data.meshId;
	materialId = data.materialId;
}