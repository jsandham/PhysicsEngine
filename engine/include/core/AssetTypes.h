#ifndef ASSET_TYPES_H__
#define ASSET_TYPES_H__

#include "Types.h"

#include "Cubemap.h"
#include "Mesh.h"
#include "Material.h"
#include "RenderTexture.h"
#include "Texture2D.h"
#include "Shader.h"

namespace PhysicsEngine
{
    template <typename T> struct AssetType
    {
        static constexpr int type = PhysicsEngine::INVALID_TYPE;
    };

    template <> struct AssetType<Cubemap>
    {
        static constexpr int type = PhysicsEngine::CUBEMAP_TYPE;
    };

    template <> struct AssetType<Mesh>
    {
        static constexpr int type = PhysicsEngine::MESH_TYPE;
    };

    template <> struct AssetType<Material>
    {
        static constexpr int type = PhysicsEngine::MATERIAL_TYPE;
    };

    template <> struct AssetType<RenderTexture>
    {
        static constexpr int type = PhysicsEngine::RENDER_TEXTURE_TYPE;
    };

    template <> struct AssetType<Texture2D>
    {
        static constexpr int type = PhysicsEngine::TEXTURE2D_TYPE;
    };

    template <> struct AssetType<Shader>
    {
        static constexpr int type = PhysicsEngine::SHADER_TYPE;
    };
}

#endif