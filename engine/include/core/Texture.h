#ifndef TEXTURE_H__
#define TEXTURE_H__

#include "Asset.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
enum class TextureFormat
{
    Depth = 0,
    RG = 1,
    RGB = 2,
    RGBA = 3
};

enum class TextureWrapMode
{
    // D3D11: D3D11_TEXTURE_ADDRESS_WRAP. OpenG: GL_REPEAT
    Repeat = 0,
    // D3D11: D3D11_TEXTURE_ADDRESS_CLAMP. OpenGL: GL_CLAMP_TO_EDGE
    ClampToEdge = 1,
    // D3D11: D3D11_TEXTURE_ADDRESS_BORDER. OpenGL: GL_CLAMP_TO_BORDER
    ClampToBorder = 2,
    // D3D11: D3D11_TEXTURE_ADDRESS_MIRROR. OpenGL: GL_MIRRORED_REPEAT
    MirrorRepeat = 3,
    // D3D11: D3D11_TEXTURE_ADDRESS_MIRROR_ONCE. OpenGL: GL_MIRROR_CLAMP_TO_EDGE
    MirrorClampToEdge = 4
};

enum class TextureFilterMode
{
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2
};

enum class TextureDimension
{
    Tex2D = 0,
    Cube = 1
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

    bool mDeviceUpdateRequired;
    bool mUpdateRequired;

  public:
    Texture(World *world, const Id &id);
    Texture(World *world, const Guid &guid, const Id &id);
    ~Texture();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual void copyTextureToDevice() = 0;
    virtual void updateTextureParameters() = 0;
    virtual void readPixels() = 0;
    virtual void writePixels() = 0;

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