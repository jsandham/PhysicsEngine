#ifndef __CUBEMAP_H__
#define __CUBEMAP_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct CubemapHeader
{
    Guid mTextureId;
    size_t mTextureSize;
    int32_t mWidth;
    int32_t mNumChannels;
    uint8_t mDimension;
    uint8_t mFormat;
};
#pragma pack(pop)

typedef enum CubemapFace
{
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ
} CubemapFace;

class Cubemap : public Texture
{
  private:
    int mWidth;

  public:
    Cubemap();
    Cubemap(const std::vector<char> &data);
    Cubemap(int width);
    Cubemap(int width, TextureFormat format);
    Cubemap(int width, int height, TextureFormat format);
    ~Cubemap();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid assetId) const;
    void deserialize(const std::vector<char> &data);

    int getWidth() const;

    std::vector<unsigned char> getRawCubemapData() const;
    std::vector<Color32> getPixels(CubemapFace face) const;
    Color32 getPixel(CubemapFace face, int x, int y) const;

    void setRawCubemapData(std::vector<unsigned char> data);
    void setPixels(CubemapFace face, int x, int y, Color32 color);
    void setPixel(CubemapFace face, int x, int y, Color32 color);

    void create();
    void destroy();
    void readPixels();
    void apply();
};

template <typename T> struct IsCubemap
{
    static constexpr bool value = false;
};

template <> struct AssetType<Cubemap>
{
    static constexpr int type = PhysicsEngine::CUBEMAP_TYPE;
};
template <> struct IsTexture<Cubemap>
{
    static constexpr bool value = true;
};
template <> struct IsCubemap<Cubemap>
{
    static constexpr bool value = true;
};
template <> struct IsAsset<Cubemap>
{
    static constexpr bool value = true;
};
template <> struct IsAssetInternal<Cubemap>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif