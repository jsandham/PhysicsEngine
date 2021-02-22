#ifndef __TEXTURE3D_H__
#define __TEXTURE3D_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
class Texture3D : public Texture
{
  private:
    int mWidth;
    int mHeight;
    int mDepth;

  public:
    Texture3D();
    Texture3D(Guid id);
    Texture3D(int width, int height, int depth, int numChannels);
    ~Texture3D();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

    int getWidth() const;
    int getHeight() const;
    int getDepth() const;

    void redefine(int width, int height, int depth, TextureFormat format);

    std::vector<unsigned char> getRawTextureData() const;
    Color getPixel(int x, int y, int z) const;

    void setRawTextureData(const std::vector<unsigned char> &data, int width, int height, int depth,
                           TextureFormat format);
    void setPixel(int x, int y, int z, const Color &color);

    void create() override;
    void destroy() override;
    void update() override;
    void readPixels() override;
    void writePixels() override;
};

template <> struct AssetType<Texture3D>
{
    static constexpr int type = PhysicsEngine::TEXTURE3D_TYPE;
};

template <> struct IsAssetInternal<Texture3D>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif