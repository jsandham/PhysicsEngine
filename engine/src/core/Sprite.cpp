#include "../../include/core/Sprite.h"
#include "../../include/core/World.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Sprite::Sprite(World *world) : Asset(world)
{
    mCreated = false;
    mChanged = false;

    mPixelsPerUnit = 100;
}

Sprite::Sprite(World *world, Id id) : Asset(world, id)
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

    out["textureId"] = mWorld->getGuidOf(mTextureId);
    out["pixelsPerUnit"] = mPixelsPerUnit;
}

void Sprite::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mTextureId = mWorld->getIdOf(YAML::getValue<Guid>(in, "textureId"));
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

Id Sprite::getTextureId() const
{
    return mTextureId;
}

void Sprite::setTextureId(Id textureId)
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

    Graphics::createSprite(&mVao);

    mCreated = true;
}

void Sprite::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroySprite(&mVao);

    mCreated = false;
}