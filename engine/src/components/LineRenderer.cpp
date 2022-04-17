#include "../../include/components/LineRenderer.h"

#include "../../include/core/GLM.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer(World* world) : Component(world)
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);
    mEnabled = true;

    mMaterialId = -1;
}

LineRenderer::LineRenderer(World* world, Id id) : Component(world, id)
{
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);
    mEnabled = true;

    mMaterialId = -1;
}

LineRenderer::~LineRenderer()
{
}

void LineRenderer::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["materialId"] = mWorld->getGuidOf(mMaterialId);
    out["start"] = mStart;
    out["end"] = mEnd;
    out["enabled"] = mEnabled;

}

void LineRenderer::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mMaterialId = mWorld->getIdOf(YAML::getValue<Guid>(in, "materialId"));
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