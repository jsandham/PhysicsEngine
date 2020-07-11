#include "../../include/components/LineRenderer.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer()
{
	mStart = glm::vec3(0.0f, 0.0f, 0.0f);
	mEnd = glm::vec3(1.0f, 0.0f, 0.0f);

	mMaterialId = Guid::INVALID;
}

LineRenderer::LineRenderer(const std::vector<char>& data)
{
	deserialize(data);
}

LineRenderer::~LineRenderer()
{

}

std::vector<char> LineRenderer::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> LineRenderer::serialize(Guid componentId, Guid entityId) const
{
	LineRendererHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mStart = mStart;
	header.mEnd = mEnd;
	header.mMaterialId = mMaterialId;

	std::vector<char> data(sizeof(LineRendererHeader));

	memcpy(&data[0], &header, sizeof(LineRendererHeader));

	return data;
}

void LineRenderer::deserialize(const std::vector<char>& data)
{
	const LineRendererHeader* header = reinterpret_cast<const LineRendererHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mStart = header->mStart;
	mEnd = header->mEnd;
	mMaterialId = header->mMaterialId;
}