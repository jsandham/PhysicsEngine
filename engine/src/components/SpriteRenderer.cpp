#include "../../include/components/SpriteRenderer.h"

#include "../../include/core/InternalShaders.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

SpriteRenderer::SpriteRenderer() : Component()
{
    mSpriteId = Guid::INVALID;
    mColor = Color::white;
    mSpriteChanged = true;
    mIsStatic = true;
    mEnabled = true;
}

SpriteRenderer::SpriteRenderer(Guid id) : Component(id)
{
    mSpriteId = Guid::INVALID;
    mColor = Color::white;
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
    out["isStatic"] = mIsStatic;
    out["enabled"] = mEnabled;
}

void SpriteRenderer::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mSpriteId = YAML::getValue<Guid>(in, "spriteId");
    mColor = YAML::getValue<Color>(in, "color");
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

//void SpriteRenderer::init()
//{
//    Graphics::compile(InternalShaders::spriteVertexShader, InternalShaders::spriteFragmentShader, "", &mShader);
//
//    Graphics::use(mShader);
//
//    mModelLoc = Graphics::findUniformLocation("model", mShader);
//    mProjectionLoc = Graphics::findUniformLocation("projection", mShader);
//    mSpriteColorLoc = Graphics::findUniformLocation("spriteColor", mShader);
//    mImageLoc = Graphics::findUniformLocation("image", mShader);
//
//    assert(mModelLoc != -1);
//    assert(mProjectionLoc != -1);
//    assert(mSpriteColorLoc != -1);
//    assert(mImageLoc != -1);
//
//    Graphics::createSprite(&mVAO);
//}
//
//void SpriteRenderer::drawSprite(Camera* camera, GLint texture, const glm::vec2& position, const glm::vec2& size, float rotate)
//{
//    Graphics::use(mShader);
//    glm::mat4 model = glm::mat4(1.0f);
//    model = glm::translate(model, glm::vec3(position, 0.0f));
//    model = glm::translate(model, glm::vec3(0.5f * size.x, 0.5f * size.y, 0.0f));
//    model = glm::rotate(model, glm::radians(rotate), glm::vec3(0.0f, 0.0f, 1.0f));
//    model = glm::translate(model, glm::vec3(-0.5f * size.x, -0.5f * size.y, 0.0f));
//    model = glm::scale(model, glm::vec3(size, 1.0f));
//
//    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(camera->getViewport().mWidth), 0.0f, static_cast<float>(camera->getViewport().mHeight), -1.0f, 1.0f);
//
//    Graphics::setMat4(mModelLoc, model);
//    Graphics::setMat4(mProjectionLoc, projection);
//    Graphics::setVec3(mSpriteColorLoc, glm::vec3(1, 1, 1));
//    Graphics::setTexture2D(mImageLoc, 0, texture);
//
//    float width = camera->getViewport().mWidth;
//    float height = camera->getViewport().mHeight;
//
//    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
//    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
//        camera->getViewport().mHeight);
//
//    Graphics::render(0, 6, mVAO);
//
//    Graphics::unbindFramebuffer();
//}
