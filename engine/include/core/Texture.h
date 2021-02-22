#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "Asset.h"
#include "Guid.h"

namespace PhysicsEngine
{
typedef enum TextureDimension
{
    Tex2D = 0,
    Tex3D = 1,
    Cube = 2
} TextureDimension;

typedef enum TextureFormat
{
    Depth = 0,
    RG = 1,
    RGB = 2,
    RGBA = 3
} TextureFormat;

typedef enum TextureWrapMode
{
    Repeat = 0,
    Clamp = 1
} TextureWrapMode;

typedef enum TextureFilterMode
{
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2
} TextureFilterMode;

class Texture : public Asset
{
  protected:
    std::vector<unsigned char> mRawTextureData;
    int mNumChannels;
    int mAnisoLevel;
    TextureDimension mDimension;
    TextureFormat mFormat;
    TextureWrapMode mWrapMode;
    TextureFilterMode mFilterMode;
    GLuint mTex;
    bool mCreated;
    bool mUpdateRequired;

  public:
    Texture();
    Texture(Guid id);
    ~Texture();

    virtual void serialize(std::ostream &out) const;
    virtual void deserialize(std::istream &in);

    virtual void create() = 0;
    virtual void destroy() = 0;
    virtual void update() = 0;
    virtual void readPixels() = 0;
    virtual void writePixels() = 0;

    bool isCreated() const;
    bool updateRequired() const;
    int getNumChannels() const;
    int getAnisoLevel() const;
    TextureDimension getDimension() const;
    TextureFormat getFormat() const;
    TextureWrapMode getWrapMode() const;
    TextureFilterMode getFilterMode() const;
    GLuint getNativeGraphics() const;

    void setAnisoLevel(int anisoLevel);
    void setWrapMode(TextureWrapMode wrapMode);
    void setFilterMode(TextureFilterMode filterMode);

  protected:
    int calcNumChannels(TextureFormat format) const;
};

template <> struct IsAssetInternal<Texture>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif