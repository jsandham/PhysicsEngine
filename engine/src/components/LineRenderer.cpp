#include "../../include/components/LineRenderer.h"

#include "../../include/core/Serialize.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer() : Component()
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);

    mMaterialId = Guid::INVALID;
}

LineRenderer::LineRenderer(Guid id) : Component(id)
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);

    mMaterialId = Guid::INVALID;
}

LineRenderer::~LineRenderer()
{
}

std::vector<char> LineRenderer::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> LineRenderer::serialize(const Guid &componentId, const Guid &entityId) const
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

void LineRenderer::deserialize(const std::vector<char> &data)
{
    const LineRendererHeader *header = reinterpret_cast<const LineRendererHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mStart = header->mStart;
    mEnd = header->mEnd;
    mMaterialId = header->mMaterialId;
}

void LineRenderer::serialize(std::ostream& out) const
{
    Component::serialize(out);

    PhysicsEngine::write<Guid>(out, mMaterialId);
    PhysicsEngine::write<glm::vec3>(out, mStart);
    PhysicsEngine::write<glm::vec3>(out, mEnd);
}

void LineRenderer::deserialize(std::istream& in)
{
    Component::deserialize(in);

    PhysicsEngine::read<Guid>(in, mMaterialId);
    PhysicsEngine::read<glm::vec3>(in, mStart);
    PhysicsEngine::read<glm::vec3>(in, mEnd);
}