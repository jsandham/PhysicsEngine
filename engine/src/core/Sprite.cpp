#include "../../include/core/Sprite.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Sprite::Sprite() : Asset()
{
    mCreated = false;
    mChanged = false;
}

Sprite::Sprite(Guid id) : Asset(id)
{
    mCreated = false;
    mChanged = false;
}

Sprite::~Sprite()
{

}

void Sprite::serialize(YAML::Node& out) const
{
    Asset::serialize(out);

    out["textureId"] = mTextureId;
}

void Sprite::deserialize(const YAML::Node& in)
{
    Asset::deserialize(in);

    mTextureId = YAML::getValue<Guid>(in, "textureId");
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

GLuint Sprite::getNativeGraphicsVAO() const
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