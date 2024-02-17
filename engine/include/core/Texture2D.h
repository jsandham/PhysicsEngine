#ifndef TEXTURE2D_H__
#define TEXTURE2D_H__

#include <string>
#include <vector>

#include "SerializationEnums.h"
#include "AssetEnums.h"
#include "Color.h"
#include "Guid.h"
#include "Id.h"

#include "../graphics/TextureHandle.h"

namespace PhysicsEngine
{
class World;

class Texture2D
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
    
    std::string mSource;
    int mWidth;
    int mHeight;

    TextureHandle *mTex;

    friend class World;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    Texture2D(World *world, const Id &id);
    Texture2D(World *world, const Guid &guid, const Id &id);
    Texture2D(World *world, const Id &id, int width, int height);
    Texture2D(World *world, const Id &id, int width, int height, TextureFormat format);
    ~Texture2D();
    Texture2D &operator=(Texture2D &&other);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    bool writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void load(const std::string &filepath);
    void writeToPNG(const std::string &filepath) const;
    void writeToJPG(const std::string &filepath) const;
    void writeToBMP(const std::string &filepath) const;

    int getWidth() const;
    int getHeight() const;

    void redefine(int width, int height, TextureFormat format);

    std::vector<unsigned char> getRawTextureData() const;
    std::vector<Color32> getPixels() const;
    Color32 getPixel(int x, int y) const;

    void setRawTextureData(const std::vector<unsigned char> &data, int width, int height, TextureFormat format);
    void setPixels(const std::vector<Color32> &colors);
    void setPixel(int x, int y, const Color32 &color);

    void copyTextureToDevice();
    void updateTextureParameters();
    void readPixels();
    void writePixels();






    bool deviceUpdateRequired() const;
    bool updateRequired() const;
    int getNumChannels() const;
    int getAnisoLevel() const;
    TextureDimension getDimension() const;
    TextureFormat getFormat() const;
    TextureWrapMode getWrapMode() const;
    TextureFilterMode getFilterMode() const;

    void setAnisoLevel(int anisoLevel);
    void setWrapMode(TextureWrapMode wrapMode);
    void setFilterMode(TextureFilterMode filterMode);







    TextureHandle *getNativeGraphics() const;
};

} // namespace PhysicsEngine

#endif