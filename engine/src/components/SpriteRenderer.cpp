#include "../../include/components/SpriteRenderer.h"

#include "../../include/core/InternalShaders.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

SpriteRenderer::SpriteRenderer() : Component()
{
    mSpriteId = Guid::INVALID;
    mColor = Color::white;
    mFlipX = false;
    mFlipY = false;
    mSpriteChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

SpriteRenderer::SpriteRenderer(Guid id) : Component(id)
{
    mSpriteId = Guid::INVALID;
    mColor = Color::white;
    mFlipX = false;
    mFlipY = false;
    mSpriteChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

SpriteRenderer::~SpriteRenderer()
{
    
}

void SpriteRenderer::serialize(YAML::Node& out) const
{
    Component::serialize(out);

    out["spriteId"] = mSpriteId;
    out["color"] = mColor;
    out["flipX"] = mFlipX;
    out["flipY"] = mFlipY;
    out["isStatic"] = mIsStatic;
    out["enabled"] = mEnabled;
}

void SpriteRenderer::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mSpriteId = YAML::getValue<Guid>(in, "spriteId");
    mColor = YAML::getValue<Color>(in, "color");
    mFlipX = YAML::getValue<bool>(in, "flipX");
    mFlipY = YAML::getValue<bool>(in, "flipY");
    mIsStatic = YAML::getValue<bool>(in, "isStatic");
    mEnabled = YAML::getValue<bool>(in, "enabled");

    mSpriteChanged = true;
}

int SpriteRenderer::getType() const
{
    return PhysicsEngine::SPRITERENDERER_TYPE;
}

std::string SpriteRenderer::getObjectName() const
{
    return PhysicsEngine::SPRITERENDERER_NAME;
}

void SpriteRenderer::setSprite(Guid id)
{
    mSpriteId = id;
    mSpriteChanged = true;
}

Guid SpriteRenderer::getSprite() const
{
    return mSpriteId;
}
