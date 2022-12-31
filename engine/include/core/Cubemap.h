#ifndef CUBEMAP_H__
#define CUBEMAP_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

#include "../graphics/CubemapHandle.h"

namespace PhysicsEngine
{
enum class CubemapFace
{
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ
};

class Cubemap : public Texture
{
  private:
    Guid mLeftTexGuid;
    Guid mRightTexGuid;
    Guid mBottomTexGuid;
    Guid mTopTexGuid;
    Guid mBackTexGuid;
    Guid mFrontTexGuid;

    int mWidth;

    CubemapHandle *mCube;

  public:
    Cubemap(World *world, const Id &id);
    Cubemap(World *world, const Guid &guid, const Id &id);
    Cubemap(World *world, const Id &id, int width);
    Cubemap(World *world, const Id &id, int width, TextureFormat format);
    ~Cubemap();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    int getWidth() const;

    std::vector<unsigned char> getRawCubemapData() const;
    std::vector<Color32> getPixels(CubemapFace face) const;
    Color32 getPixel(CubemapFace face, int x, int y) const;
    Guid getTexId(CubemapFace face) const;

    void setRawCubemapData(const std::vector<unsigned char> &data);
    void setRawCubemapData(CubemapFace face, const std::vector<unsigned char> &data);
    void setPixels(CubemapFace face, int x, int y, const Color32 &color);
    void setPixel(CubemapFace face, int x, int y, const Color32 &color);
    void setTexId(CubemapFace face, const Guid& texId);

    void fillCubemapFromAttachedTexture(CubemapFace face);

    void copyTextureToDevice() override;
    void updateTextureParameters() override;
    void readPixels() override;
    void writePixels() override;

    CubemapHandle *getNativeGraphics() const;
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