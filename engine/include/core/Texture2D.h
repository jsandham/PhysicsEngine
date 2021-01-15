#ifndef __TEXTURE2D_H__
#define __TEXTURE2D_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct Texture2DHeader
{
    Guid mTextureId;
    char mTextureName[64];
    size_t mTextureSize;
    int32_t mWidth;
    int32_t mHeight;
    int32_t mNumChannels;
    int32_t mAnisoLevel;
    uint8_t mDimension;
    uint8_t mFormat;
    uint8_t mWrapMode;
    uint8_t mFilterMode;
};
#pragma pack(pop)

class Texture2D : public Texture
{
  private:
    int mWidth;
    int mHeight;

  public:
    Texture2D();
    Texture2D(Guid id);
    Texture2D(int width, int height);
    Texture2D(int width, int height, TextureFormat format);
    ~Texture2D();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid assetId) const;
    void deserialize(const std::vector<char> &data);

    void load(const std::string &filepath);

    int getWidth() const;
    int getHeight() const;

    void redefine(int width, int height, TextureFormat format);

    std::vector<unsigned char> getRawTextureData() const;
    std::vector<Color32> getPixels() const;
    Color32 getPixel(int x, int y) const;

    void setRawTextureData(const std::vector<unsigned char> &data, int width, int height, TextureFormat format);
    void setPixels(const std::vector<Color32> &colors);
    void setPixel(int x, int y, const Color32 &color);

    void create() override;
    void destroy() override;
    void update() override;
    void readPixels() override;
    void writePixels() override;
};

template <typename T> struct IsTexture2D
{
    static constexpr bool value = false;
};

template <> struct AssetType<Texture2D>
{
    static constexpr int type = PhysicsEngine::TEXTURE2D_TYPE;
};
template <> struct IsTexture<Texture2D>
{
    static constexpr bool value = true;
};
template <> struct IsTexture2D<Texture2D>
{
    static constexpr bool value = true;
};
template <> struct IsAsset<Texture2D>
{
    static constexpr bool value = true;
};
template <> struct IsAssetInternal<Texture2D>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif