#ifndef TEXTURE2D_H__
#define TEXTURE2D_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
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

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const std::string &filepath);
    void writeToPNG(const std::string& filepath) const;
    void writeToJPG(const std::string& filepath) const;
    void writeToBMP(const std::string& filepath) const;

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

template <> struct AssetType<Texture2D>
{
    static constexpr int type = PhysicsEngine::TEXTURE2D_TYPE;
};

template <> struct IsAssetInternal<Texture2D>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif