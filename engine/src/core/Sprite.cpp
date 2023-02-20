#include "../../include/core/Sprite.h"

#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Sprite::Sprite(World *world, const Id &id) : Asset(world, id)
{
    mBuffer = VertexBuffer::create();
    mHandle = MeshHandle::create();

    mHandle->addVertexBuffer(mBuffer, AttribType::Vec4);

    float vertices[] = {// pos      // tex
                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    mBuffer->bind();
    if (mBuffer->getSize() < sizeof(float) * 24)
    {
        mBuffer->resize(sizeof(float) * 24);
    }
    mBuffer->setData(vertices, 0, sizeof(float) * 24);
    mBuffer->unbind();

    mPixelsPerUnit = 100;
    mChanged = false;
}

Sprite::Sprite(World *world, const Guid &guid, const Id &id) : Asset(world, guid, id)
{
    mBuffer = VertexBuffer::create();
    mHandle = MeshHandle::create();
   
    mHandle->addVertexBuffer(mBuffer, AttribType::Vec4);

    float vertices[] = {// pos      // tex
                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    mBuffer->bind();
    if (mBuffer->getSize() < sizeof(float) * 24)
    {
        mBuffer->resize(sizeof(float) * 24);
    }
    mBuffer->setData(vertices, 0, sizeof(float) * 24);
    mBuffer->unbind();

    mPixelsPerUnit = 100;
    mChanged = false;
}

Sprite::~Sprite()
{
    delete mBuffer;
    delete mHandle;
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

bool Sprite::isChanged() const
{
    return mChanged;
}

MeshHandle *Sprite::getNativeGraphicsHandle() const
{
    return mHandle;
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