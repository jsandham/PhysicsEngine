#ifndef __TEXTURE3D_H__
#define __TEXTURE3D_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct Texture3DHeader
{
    Guid mTextureId;
    size_t mTextureSize;
    int32_t mWidth;
    int32_t mHeight;
    int32_t mDepth;
    int32_t mNumChannels;
    uint8_t mDimension;
    uint8_t mFormat;
};
#pragma pack(pop)

class Texture3D : public Texture
{
  private:
    int mWidth;
    int mHeight;
    int mDepth;

  public:
    Texture3D();
    Texture3D(const std::vector<char> &data);
    Texture3D(int width, int height, int depth, int numChannels);
    ~Texture3D();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid assetId) const;
    void deserialize(const std::vector<char> &data);

    int getWidth() const;
    int getHeight() const;
    int getDepth() const;

    void redefine(int width, int height, int depth, TextureFormat format);

    std::vector<unsigned char> getRawTextureData() const;
    Color getPixel(int x, int y, int z) const;

    void setRawTextureData(std::vector<unsigned char> data, int width, int height, int depth, TextureFormat format);
    void setPixel(int x, int y, int z, Color color);

    void create();
    void destroy();
    void readPixels();
    void writePixels();
};

template <typename T> struct IsTexture3D
{
    static constexpr bool value = false;
};

template <> struct AssetType<Texture3D>
{
    static constexpr int type = PhysicsEngine::TEXTURE3D_TYPE;
};
template <> struct IsTexture<Texture3D>
{
    static constexpr bool value = true;
};
template <> struct IsTexture3D<Texture3D>
{
    static constexpr bool value = true;
};
template <> struct IsAsset<Texture3D>
{
    static constexpr bool value = true;
};
template <> struct IsAssetInternal<Texture3D>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif