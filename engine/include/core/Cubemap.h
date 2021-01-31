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
    char mTextureName[64];
    size_t mTextureSize;
    int32_t mWidth;
    int32_t mNumChannels;
    int32_t mAnisoLevel;
    uint8_t mDimension;
    uint8_t mFormat;
    uint8_t mWrapMode;
    uint8_t mFilterMode;
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
    Cubemap(Guid id);
    Cubemap(int width);
    Cubemap(int width, TextureFormat format);
    Cubemap(int width, int height, TextureFormat format);
    ~Cubemap();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid assetId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    int getWidth() const;

    std::vector<unsigned char> getRawCubemapData() const;
    std::vector<Color32> getPixels(CubemapFace face) const;
    Color32 getPixel(CubemapFace face, int x, int y) const;

    void setRawCubemapData(const std::vector<unsigned char> &data);
    void setPixels(CubemapFace face, int x, int y, const Color32 &color);
    void setPixel(CubemapFace face, int x, int y, const Color32 &color);

    void create() override;
    void destroy() override;
    void update() override;
    void readPixels() override;
    void writePixels() override;
};

template <> struct AssetType<Cubemap>
{
    static constexpr int type = PhysicsEngine::CUBEMAP_TYPE;
};
template <> struct IsAssetInternal<Cubemap>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif