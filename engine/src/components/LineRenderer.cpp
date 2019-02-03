#include "../../include/components/LineRenderer.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer()
{
	start = glm::vec3(0.0f, 0.0f, 0.0f);
	end = glm::vec3(1.0f, 0.0f, 0.0f);

	materialId = Guid::INVALID;
}

LineRenderer::LineRenderer(unsigned char* data)
{
	
}

LineRenderer::~LineRenderer()
{

}

void LineRenderer::load(LineRendererData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	start = data.start;
	end = data.end;

	materialId = data.materialId;
}