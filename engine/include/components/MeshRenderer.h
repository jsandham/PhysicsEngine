#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include <vector>

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshRendererHeader
	{
		Guid mComponentId;
		Guid mEntityId;
		Guid mMeshId;
		Guid mMaterialIds[8];
		int mMaterialCount;
		bool mIsStatic;
		bool mEnabled;
	};
#pragma pack(pop)

	class MeshRenderer : public Component
	{
		private:
			Guid mMeshId;
			Guid mMaterialIds[8];

		public:
			int mMaterialCount;
			bool mMeshChanged;
			bool mMaterialChanged;
			bool mIsStatic;
			bool mEnabled;

		public:
			MeshRenderer();
			MeshRenderer(std::vector<char> data);
			~MeshRenderer();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			void setMesh(Guid meshId);
			void setMaterial(Guid materialId);
			void setMaterial(Guid materialId, int index);

			Guid getMesh() const;
			Guid getMaterial() const;
			Guid getMaterial(int index) const;
			std::vector<Guid> getMaterials() const;
	};

	template <>
	const int ComponentType<MeshRenderer>::type = 3;

	template <typename T>
	struct IsMeshRenderer { static const bool value; };

	template <typename T>
	const bool IsMeshRenderer<T>::value = false;

	template<>
	const bool IsMeshRenderer<MeshRenderer>::value = true;
	template<>
	const bool IsComponent<MeshRenderer>::value = true;
	template<>
	const bool IsComponentInternal<MeshRenderer>::value = true;
}

#endif