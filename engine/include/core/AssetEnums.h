#ifndef ASSET_ENUMS_H__
#define ASSET_ENUMS_H__

#include <string>

namespace PhysicsEngine
{
    // Cubemap
    enum class CubemapFace
    {
        PositiveX,
        NegativeX,
        PositiveY,
        NegativeY,
        PositiveZ,
        NegativeZ
    };

    constexpr auto CubemapFaceToString(CubemapFace face)
    {
        switch (face)
        {
        case CubemapFace::NegativeX:
            return "NegativeX";
        case CubemapFace::NegativeY:
            return "NegativeY";
        case CubemapFace::NegativeZ:
            return "NegativeZ";
        case CubemapFace::PositiveX:
            return "PositiveX";
        case CubemapFace::PositiveY:
            return "PositiveY";
        case CubemapFace::PositiveZ:
            return "PositiveZ";
        }
    }

    // Textures
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

    // Shaders
    enum class RenderQueue
    {
        Opaque = 0,
        Transparent = 1
    };

    enum class ShaderMacro
    {
        None = 0,
        Directional = 1,
        Spot = 2,
        Point = 4,
        HardShadows = 8,
        SoftShadows = 16,
        SSAO = 32,
        ShowCascades = 64,
        Instancing = 128
    };

    constexpr auto RenderQueueToString(RenderQueue renderQueue)
    {
        switch (renderQueue)
        {
        case RenderQueue::Opaque:
            return "Opaque";
        case RenderQueue::Transparent:
            return "Transparent";
        }
    }

    constexpr auto ShaderMacroToString(ShaderMacro macro)
    {
        switch (macro)
        {
        case ShaderMacro::Directional:
            return "Directional";
        case ShaderMacro::Spot:
            return "Spot";
        case ShaderMacro::Point:
            return "Point";
        case ShaderMacro::HardShadows:
            return "HardShadows";
        case ShaderMacro::SoftShadows:
            return "SoftShadows";
        case ShaderMacro::SSAO:
            return "SSAO";
        case ShaderMacro::ShowCascades:
            return "ShadowCascades";
        case ShaderMacro::Instancing:
            return "Instancing";
        case ShaderMacro::None:
            return "None";
        }
    }

    enum class ShaderUniformType
    {
        Int = 0,
        Float = 1,
        Color = 2,
        Vec2 = 3,
        Vec3 = 4,
        Vec4 = 5,
        Mat2 = 6,
        Mat3 = 7,
        Mat4 = 8,
        Sampler2D = 9,
        SamplerCube = 10,
        Invalid = 11
    };

    constexpr auto ShaderUniformTypeToString(ShaderUniformType type)
    {
        switch (type)
        {
        case ShaderUniformType::Int:
            return "Int";
        case ShaderUniformType::Float:
            return "Float";
        case ShaderUniformType::Color:
            return "Color";
        case ShaderUniformType::Vec2:
            return "Vec2";
        case ShaderUniformType::Vec3:
            return "Vec3";
        case ShaderUniformType::Vec4:
            return "Vec4";
        case ShaderUniformType::Mat2:
            return "Mat2";
        case ShaderUniformType::Mat3:
            return "Mat3";
        case ShaderUniformType::Mat4:
            return "Mat4";
        case ShaderUniformType::Sampler2D:
            return "Sampler2D";
        case ShaderUniformType::SamplerCube:
            return "SamplerCube";
        case ShaderUniformType::Invalid:
            return "Invalid";
        }

        return "Invalid";
    }





    struct ShaderUniform
    {
        std::string mBufferName; // "Block name" in OpenGL
        std::string mName;       // variable name (including block name if applicable)
        ShaderUniformType mType; // type of the uniform (float, vec3 or mat4, etc)
        int mUniformId;          // integer hash of uniform name

        char mData[64];
        void *mTex; // if data stores a texture id, this is the texture handle

        std::string getShortName() const //getDisplayName?
        {
            size_t pos = mName.find_first_of('.');
            return mName.substr(pos + 1);
        }
    };
}

#endif