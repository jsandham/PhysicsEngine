#ifndef RENDER_TEXTURE_H__
#define RENDER_TEXTURE_H__

#include "../graphics/Framebuffer.h"
#include "Texture.h"

namespace PhysicsEngine
{
struct RenderTextureTargets
{
    Framebuffer *mMainFBO;
};

class RenderTexture : public Texture
{
  private:
    RenderTextureTargets mTargets;
    int mWidth;
    int mHeight;

  public:
    RenderTexture(World *world, const Id &id);
    RenderTexture(World *world, const Guid &guid, const Id &id);
    RenderTexture(World *world, const Id &id, int width, int height);
    RenderTexture(World *world, const Id &id, int width, int height, TextureFormat format);
    ~RenderTexture();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void writeToPNG(const std::string &filepath) const;
    void writeToJPG(const std::string &filepath) const;
    void writeToBMP(const std::string &filepath) const;

    int getWidth() const;
    int getHeight() const;

    void copyTextureToDevice() override;
    void updateTextureParameters() override;
    void readPixels() override;
    void writePixels() override;

    Framebuffer *getNativeGraphicsMainFBO() const;
    RenderTextureHandle *getNativeGraphicsColorTex() const;
    RenderTextureHandle *getNativeGraphicsDepthTex() const;
};

template <> struct AssetType<RenderTexture>
{
    static constexpr int type = PhysicsEngine::RENDER_TEXTURE_TYPE;
};

template <> struct IsAssetInternal<RenderTexture>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif