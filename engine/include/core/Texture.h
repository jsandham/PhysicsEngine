#ifndef TEXTURE_H__
#define TEXTURE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "Asset.h"
#include "Guid.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
enum class TextureDimension
{
    Tex2D = 0,
    Tex3D = 1,
    Cube = 2
};

enum class TextureFormat
{
    Depth = 0,
    RG = 1,
    RGB = 2,
    RGBA = 3
};

enum class TextureWrapMode
{
    Repeat = 0,
    Clamp = 1
};

enum class TextureFilterMode
{
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2
};

class Texture : public Asset
{
  protected:
    std::vector<unsigned char> mRawTextureData;
    int mNumChannels;
    int mAnisoLevel;
    TextureDimension mDimension;
    TextureFormat mFormat;
    TextureWrapMode mWrapMode;
    TextureFilterMode mFilterMode;
    GLuint mTex;
    bool mCreated;
    bool mUpdateRequired;

  public:
    Texture(World* world);
    Texture(World* world, Guid id);
    ~Texture();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual void create() = 0;
    virtual void destroy() = 0;
    virtual void update() = 0;
    virtual void readPixels() = 0;
    virtual void writePixels() = 0;

    bool isCreated() const;
    bool updateRequired() const;
    int getNumChannels() const;
    int getAnisoLevel() const;
    TextureDimension getDimension() const;
    TextureFormat getFormat() const;
    TextureWrapMode getWrapMode() const;
    TextureFilterMode getFilterMode() const;
    GLuint getNativeGraphics() const;

    void setAnisoLevel(int anisoLevel);
    void setWrapMode(TextureWrapMode wrapMode);
    void setFilterMode(TextureFilterMode filterMode);

  protected:
    int calcNumChannels(TextureFormat format) const;
};

template <> struct IsAssetInternal<Texture>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

namespace YAML
{
// TextureDimension
template <> struct convert<PhysicsEngine::TextureDimension>
{
    static Node encode(const PhysicsEngine::TextureDimension &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::TextureDimension &rhs)
    {
        rhs = static_cast<PhysicsEngine::TextureDimension>(node.as<int>());
        return true;
    }
};

// TextureFormat
template <> struct convert<PhysicsEngine::TextureFormat>
{
    static Node encode(const PhysicsEngine::TextureFormat &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::TextureFormat &rhs)
    {
        rhs = static_cast<PhysicsEngine::TextureFormat>(node.as<int>());
        return true;
    }
};

// TextureWrapMode
template <> struct convert<PhysicsEngine::TextureWrapMode>
{
    static Node encode(const PhysicsEngine::TextureWrapMode &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::TextureWrapMode &rhs)
    {
        rhs = static_cast<PhysicsEngine::TextureWrapMode>(node.as<int>());
        return true;
    }
};

// TextureFilterMode
template <> struct convert<PhysicsEngine::TextureFilterMode>
{
    static Node encode(const PhysicsEngine::TextureFilterMode &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::TextureFilterMode &rhs)
    {
        rhs = static_cast<PhysicsEngine::TextureFilterMode>(node.as<int>());
        return true;
    }
};
} // namespace YAML

#endif