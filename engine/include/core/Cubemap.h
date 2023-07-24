#ifndef CUBEMAP_H__
#define CUBEMAP_H__

#include <vector>

#include "SerializationEnums.h"
#include "Guid.h"
#include "Id.h"
#include "Color.h"

#include "../graphics/CubemapHandle.h"

namespace PhysicsEngine
{
class World;

class Cubemap
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

    Guid mLeftTexGuid;
    Guid mRightTexGuid;
    Guid mBottomTexGuid;
    Guid mTopTexGuid;
    Guid mBackTexGuid;
    Guid mFrontTexGuid;

    int mWidth;

    CubemapHandle *mCube;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    Cubemap(World *world, const Id &id);
    Cubemap(World *world, const Guid &guid, const Id &id);
    Cubemap(World *world, const Id &id, int width);
    Cubemap(World *world, const Id &id, int width, TextureFormat format);
    ~Cubemap();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    bool writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    int getWidth() const;

    std::vector<unsigned char> getRawCubemapData() const;
    std::vector<Color32> getPixels(CubemapFace face) const;
    Color32 getPixel(CubemapFace face, int x, int y) const;
    Guid getTexId(CubemapFace face) const;

    void setRawCubemapData(const std::vector<unsigned char> &data);
    void setRawCubemapData(CubemapFace face, const std::vector<unsigned char> &data);
    void setPixels(CubemapFace face, int x, int y, const Color32 &color);
    void setPixel(CubemapFace face, int x, int y, const Color32 &color);
    void setTexId(CubemapFace face, const Guid &texId);

    void fillCubemapFromAttachedTexture(CubemapFace face);

    void copyTextureToDevice();
    void updateTextureParameters();
    void readPixels();
    void writePixels();

    CubemapHandle *getNativeGraphics() const;
};

} // namespace PhysicsEngine

#endif