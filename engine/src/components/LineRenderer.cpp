#include "../../include/components/LineRenderer.h"

#include "../../include/core/GLM.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer(World *world, const Id &id) : Component(world, id)
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);
    mEnabled = true;

    mMaterialId = Guid::INVALID;
}

LineRenderer::LineRenderer(World *world, const Guid &guid, const Id &id) : Component(world, guid, id)
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);
    mEnabled = true;

    mMaterialId = Guid::INVALID;
}

LineRenderer::~LineRenderer()
{
}

void LineRenderer::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["materialId"] = mMaterialId;
    out["start"] = mStart;
    out["end"] = mEnd;
    out["enabled"] = mEnabled;
}

void LineRenderer::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mMaterialId = YAML::getValue<Guid>(in, "materialId");
    mStart = YAML::getValue<glm::vec3>(in, "start");
    mEnd = YAML::getValue<glm::vec3>(in, "end");
    mEnabled = YAML::getValue<bool>(in, "enabled");
}

int LineRenderer::getType() const
{
    return PhysicsEngine::LINERENDERER_TYPE;
}

std::string LineRenderer::getObjectName() const
{
    return PhysicsEngine::LINERENDERER_NAME;
}