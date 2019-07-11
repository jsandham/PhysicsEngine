#include "../../include/components/LineRenderer.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer()
{
	start = glm::vec3(0.0f, 0.0f, 0.0f);
	end = glm::vec3(1.0f, 0.0f, 0.0f);

	materialId = Guid::INVALID;
}

LineRenderer::LineRenderer(std::vector<char> data)
{
	deserialize(data);
}

LineRenderer::~LineRenderer()
{

}

std::vector<char> LineRenderer::serialize()
{
	LineRendererHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.start = start;
	header.end = end;
	header.materialId = materialId;

	int numberOfBytes = sizeof(LineRendererHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(LineRendererHeader));

	return data;
}

void LineRenderer::deserialize(std::vector<char> data)
{
	LineRendererHeader* header = reinterpret_cast<LineRendererHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	start = header->start;
	end = header->end;
	materialId = header->materialId;
}