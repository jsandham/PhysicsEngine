#include "../../include/core/Sprite.h"

#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Sprite::Sprite(World *world, const Id &id) : Asset(world, id)
{
    mCreated = false;
    mChanged = false;

    mPixelsPerUnit = 100;
}

Sprite::Sprite(World *world, const Guid &guid, const Id &id) : Asset(world, guid, id)
{
    mCreated = false;
    mChanged = false;

    mPixelsPerUnit = 100;
}

Sprite::~Sprite()
{
}

void Sprite::serialize(YAML::Node &out) const
{
    Asset::serialize(out);

    out["textureId"] = mTextureId;
    out["pixelsPerUnit"] = mPixelsPerUnit;
}

void Sprite::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mTextureId = YAML::getValue<Guid>(in, "textureId");
    mPixelsPerUnit = YAML::getValue<int>(in, "pixelsPerUnit");
}

int Sprite::getType() const
{
    return PhysicsEngine::SPRITE_TYPE;
}

std::string Sprite::getObjectName() const
{
    return PhysicsEngine::SPRITE_NAME;
}

bool Sprite::isCreated() const
{
    return mCreated;
}

bool Sprite::isChanged() const
{
    return mChanged;
}

unsigned int Sprite::getNativeGraphicsVAO() const
{
    return mVao;
}

Guid Sprite::getTextureId() const
{
    return mTextureId;
}

void Sprite::setTextureId(Guid textureId)
{
    mTextureId = textureId;

    mChanged = true;
}

void Sprite::create()
{
    if (mCreated)
    {
        return;
    }

    Renderer::getRenderer()->createSprite(&mVao);

    mCreated = true;
}

void Sprite::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Renderer::getRenderer()->destroySprite(&mVao);

    mCreated = false;
}