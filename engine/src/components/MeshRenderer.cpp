#include "../../include/components/MeshRenderer.h"

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

void MeshRenderer::load(MeshRendererData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	meshId = data.meshId;
	materialId = data.materialId;
}