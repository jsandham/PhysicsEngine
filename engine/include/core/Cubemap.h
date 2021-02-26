#ifndef CUBEMAP_H__
#define CUBEMAP_H__

#include <vector>

#include "Color.h"
#include "Texture.h"

namespace PhysicsEngine
{
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
    ~Cubemap();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

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