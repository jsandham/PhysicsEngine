#ifndef __SERIALIZATION_INTERNAL_H__
#define __SERIALIZATION_INTERNAL_H__

#include "Shader.h"
#include "Mesh.h"
#include "Material.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "Cubemap.h"
#include "Font.h"

namespace PhysicsEngine
{
	template<class T>
	Guid ExtactInternalAssetId(std::vector<char> data)
	{
		static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

		return Guid::INVALID;
	}

	template<class T>
	Guid ExtactInternalComponentId(std::vector<char> data)
	{
		static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

		return Guid::INVALID;
	}

	template<class T>
	Guid ExtactInternalSystemId(std::vector<char> data)
	{
		static_assert(IsSystem<T>::value == true, "'T' is not of type System");

		return Guid::INVALID;
	}

	// Explicit asset template specializations

	template<>
	inline Guid ExtactInternalAssetId<Shader>(std::vector<char> data)
	{
		ShaderHeader* header = reinterpret_cast<ShaderHeader*>(&data[0]);

		return header->mShaderId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Mesh>(std::vector<char> data)
	{
		MeshHeader* header = reinterpret_cast<MeshHeader*>(&data[0]);

		return header->mMeshId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Material>(std::vector<char> data)
	{
		MaterialHeader* header = reinterpret_cast<MaterialHeader*>(&data[0]);

		return header->mMaterialId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Texture2D>(std::vector<char> data)
	{
		Texture2DHeader* header = reinterpret_cast<Texture2DHeader*>(&data[0]);

		return header->mTextureId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Texture3D>(std::vector<char> data)
	{
		Texture3DHeader* header = reinterpret_cast<Texture3DHeader*>(&data[0]);

		return header->mTextureId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Cubemap>(std::vector<char> data)
	{
		CubemapHeader* header = reinterpret_cast<CubemapHeader*>(&data[0]);

		return header->mTextureId;
	}

	template<>
	inline Guid ExtactInternalAssetId<Font>(std::vector<char> data)
	{
		FontHeader* header = reinterpret_cast<FontHeader*>(&data[0]);

		return header->mFontId;
	}

	// Explicit component template specializations

	template<>
	inline Guid ExtactInternalComponentId<Transform>(std::vector<char> data)
	{
		TransformHeader* header = reinterpret_cast<TransformHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<Camera>(std::vector<char> data)
	{
		CameraHeader* header = reinterpret_cast<CameraHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<Light>(std::vector<char> data)
	{
		LightHeader* header = reinterpret_cast<LightHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<MeshRenderer>(std::vector<char> data)
	{
		MeshRendererHeader* header = reinterpret_cast<MeshRendererHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<LineRenderer>(std::vector<char> data)
	{
		LineRendererHeader* header = reinterpret_cast<LineRendererHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<Rigidbody>(std::vector<char> data)
	{
		RigidbodyHeader* header = reinterpret_cast<RigidbodyHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<SphereCollider>(std::vector<char> data)
	{
		SphereColliderHeader* header = reinterpret_cast<SphereColliderHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<BoxCollider>(std::vector<char> data)
	{
		BoxColliderHeader* header = reinterpret_cast<BoxColliderHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<CapsuleCollider>(std::vector<char> data)
	{
		CapsuleColliderHeader* header = reinterpret_cast<CapsuleColliderHeader*>(&data[0]);

		return header->mComponentId;
	}

	template<>
	inline Guid ExtactInternalComponentId<MeshCollider>(std::vector<char> data)
	{
		MeshColliderHeader* header = reinterpret_cast<MeshColliderHeader*>(&data[0]);

		return header->mComponentId;
	}

	// Explicit system template specializations

	template<>
	inline Guid ExtactInternalSystemId<RenderSystem>(std::vector<char> data)
	{
		return Guid::INVALID;
	}

	template<>
	inline Guid ExtactInternalSystemId<PhysicsSystem>(std::vector<char> data)
	{
		return Guid::INVALID;
	}

	template<>
	inline Guid ExtactInternalSystemId<CleanUpSystem>(std::vector<char> data)
	{
		return Guid::INVALID;
	}

	template<>
	inline Guid ExtactInternalSystemId<DebugSystem>(std::vector<char> data)
	{
		return Guid::INVALID;
	}
}

#endif
