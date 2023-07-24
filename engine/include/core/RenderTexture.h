#ifndef RENDER_TEXTURE_H__
#define RENDER_TEXTURE_H__

#include "../graphics/Framebuffer.h"

#include "SerializationEnums.h"
#include "AssetEnums.h"
#include "Guid.h"
#include "Id.h"

namespace PhysicsEngine
{
struct RenderTextureTargets
{
    Framebuffer *mMainFBO;
};

class World;

class RenderTexture
{
  private:
    Guid mGuid;
    Id mId;
    World *mWorld;

    std::vector<unsigned char> mRawTextureData;
    int mNumChannels;
    int mAnisoLevel;
    TextureDimension mDimension;
    TextureFormat mFormat;
    TextureWrapMode mWrapMode;
    TextureFilterMode mFilterMode;

    bool mDeviceUpdateRequired;
    bool mUpdateRequired;

    RenderTextureTargets mTargets;
    int mWidth;
    int mHeight;

    friend class World;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    RenderTexture(World *world, const Id &id);
    RenderTexture(World *world, const Guid &guid, const Id &id);
    RenderTexture(World *world, const Id &id, int width, int height);
    RenderTexture(World *world, const Id &id, int width, int height, TextureFormat format);
    ~RenderTexture();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    bool writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void writeToPNG(const std::string &filepath) const;
    void writeToJPG(const std::string &filepath) const;
    void writeToBMP(const std::string &filepath) const;

    int getWidth() const;
    int getHeight() const;

    void copyTextureToDevice();
    void updateTextureParameters();
    void readPixels();
    void writePixels();

    Framebuffer *getNativeGraphicsMainFBO() const;
    RenderTextureHandle *getNativeGraphicsColorTex() const;
    RenderTextureHandle *getNativeGraphicsDepthTex() const;
};

} // namespace PhysicsEngine

#endif