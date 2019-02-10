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
	size_t index = sizeof(int);
	index += sizeof(char);
	LineRendererHeader* header = reinterpret_cast<LineRendererHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	start = header->start;
	end = header->end;
	materialId = header->materialId;
}

LineRenderer::~LineRenderer()
{

}

void* LineRenderer::operator new(size_t size)
{
	return getAllocator<LineRenderer>().allocate();
}

void LineRenderer::operator delete(void*)
{

}