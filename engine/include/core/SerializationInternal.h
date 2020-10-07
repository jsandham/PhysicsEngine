#ifndef __SERIALIZATION_INTERNAL_H__
#define __SERIALIZATION_INTERNAL_H__

#include "Cubemap.h"
#include "Font.h"
#include "Material.h"
#include "Mesh.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"

namespace PhysicsEngine
{
template <class T> Guid ExtactInternalAssetId(const std::vector<char> &data)
{
    static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

    return Guid::INVALID;
}

template <class T> Guid ExtactInternalComponentId(const std::vector<char> &data)
{
    static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

    return Guid::INVALID;
}

template <class T> Guid ExtactInternalSystemId(const std::vector<char> &data)
{
    static_assert(IsSystem<T>::value == true, "'T' is not of type System");

    return Guid::INVALID;
}

// Explicit asset template specializations

template <> inline Guid ExtactInternalAssetId<Shader>(const std::vector<char> &data)
{
    const ShaderHeader *header = reinterpret_cast<const ShaderHeader *>(&data[0]);

    return header->mShaderId;
}

template <> inline Guid ExtactInternalAssetId<Mesh>(const std::vector<char> &data)
{
    const MeshHeader *header = reinterpret_cast<const MeshHeader *>(&data[0]);

    return header->mMeshId;
}

template <> inline Guid ExtactInternalAssetId<Material>(const std::vector<char> &data)
{
    const MaterialHeader *header = reinterpret_cast<const MaterialHeader *>(&data[0]);

    return header->mMaterialId;
}

template <> inline Guid ExtactInternalAssetId<Texture2D>(const std::vector<char> &data)
{
    const Texture2DHeader *header = reinterpret_cast<const Texture2DHeader *>(&data[0]);

    return header->mTextureId;
}

template <> inline Guid ExtactInternalAssetId<Texture3D>(const std::vector<char> &data)
{
    const Texture3DHeader *header = reinterpret_cast<const Texture3DHeader *>(&data[0]);

    return header->mTextureId;
}

template <> inline Guid ExtactInternalAssetId<Cubemap>(const std::vector<char> &data)
{
    const CubemapHeader *header = reinterpret_cast<const CubemapHeader *>(&data[0]);

    return header->mTextureId;
}

template <> inline Guid ExtactInternalAssetId<Font>(const std::vector<char> &data)
{
    const FontHeader *header = reinterpret_cast<const FontHeader *>(&data[0]);

    return header->mFontId;
}

// Explicit component template specializations

template <> inline Guid ExtactInternalComponentId<Transform>(const std::vector<char> &data)
{
    const TransformHeader *header = reinterpret_cast<const TransformHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<Camera>(const std::vector<char> &data)
{
    const CameraHeader *header = reinterpret_cast<const CameraHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<Light>(const std::vector<char> &data)
{
    const LightHeader *header = reinterpret_cast<const LightHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<MeshRenderer>(const std::vector<char> &data)
{
    const MeshRendererHeader *header = reinterpret_cast<const MeshRendererHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<LineRenderer>(const std::vector<char> &data)
{
    const LineRendererHeader *header = reinterpret_cast<const LineRendererHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<Rigidbody>(const std::vector<char> &data)
{
    const RigidbodyHeader *header = reinterpret_cast<const RigidbodyHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<SphereCollider>(const std::vector<char> &data)
{
    const SphereColliderHeader *header = reinterpret_cast<const SphereColliderHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<BoxCollider>(const std::vector<char> &data)
{
    const BoxColliderHeader *header = reinterpret_cast<const BoxColliderHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<CapsuleCollider>(const std::vector<char> &data)
{
    const CapsuleColliderHeader *header = reinterpret_cast<const CapsuleColliderHeader *>(&data[0]);

    return header->mComponentId;
}

template <> inline Guid ExtactInternalComponentId<MeshCollider>(const std::vector<char> &data)
{
    const MeshColliderHeader *header = reinterpret_cast<const MeshColliderHeader *>(&data[0]);

    return header->mComponentId;
}

// Explicit system template specializations

template <> inline Guid ExtactInternalSystemId<RenderSystem>(const std::vector<char> &data)
{
    return Guid::INVALID;
}

template <> inline Guid ExtactInternalSystemId<PhysicsSystem>(const std::vector<char> &data)
{
    return Guid::INVALID;
}

template <> inline Guid ExtactInternalSystemId<CleanUpSystem>(const std::vector<char> &data)
{
    return Guid::INVALID;
}

template <> inline Guid ExtactInternalSystemId<DebugSystem>(const std::vector<char> &data)
{
    return Guid::INVALID;
}
} // namespace PhysicsEngine

#endif
