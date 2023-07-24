#include "../../include/components/LineRenderer.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/GLM.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mStart = glm::vec3(0.0f, 0.0f, 0.0f);
    mEnd = glm::vec3(1.0f, 0.0f, 0.0f);
    mEnabled = true;

    mMaterialId = Guid::INVALID;
}

LineRenderer::LineRenderer(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
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
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["materialId"] = mMaterialId;
    out["start"] = mStart;
    out["end"] = mEnd;
    out["enabled"] = mEnabled;
}

void LineRenderer::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

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

Guid LineRenderer::getEntityGuid() const
{
    return mEntityGuid;
}

Guid LineRenderer::getGuid() const
{
    return mGuid;
}

Id LineRenderer::getId() const
{
    return mId;
}