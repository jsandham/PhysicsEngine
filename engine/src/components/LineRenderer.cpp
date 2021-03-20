#include "../../include/components/LineRenderer.h"

#include "../../include/core/Serialization.h"

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

void LineRenderer::serialize(std::ostream &out) const
{
    Component::serialize(out);

    PhysicsEngine::write<Guid>(out, mMaterialId);
    PhysicsEngine::write<glm::vec3>(out, mStart);
    PhysicsEngine::write<glm::vec3>(out, mEnd);
}

void LineRenderer::deserialize(std::istream &in)
{
    Component::deserialize(in);

    PhysicsEngine::read<Guid>(in, mMaterialId);
    PhysicsEngine::read<glm::vec3>(in, mStart);
    PhysicsEngine::read<glm::vec3>(in, mEnd);
}

void LineRenderer::serialize(YAML::Node& out) const
{
    Component::serialize(out);

    out["materialId"] = mMaterialId;
    out["start"] = mStart;
    out["end"] = mEnd;
}

void LineRenderer::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mMaterialId = YAML::getValue<Guid>(in, "materialId");
    mStart = YAML::getValue<glm::vec3>(in, "start");
    mEnd = YAML::getValue<glm::vec3>(in, "end");
}

int LineRenderer::getType() const
{
    return PhysicsEngine::LINERENDERER_TYPE;
}

std::string LineRenderer::getObjectName() const
{
    return PhysicsEngine::LINERENDERER_NAME;
}