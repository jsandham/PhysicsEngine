#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include <vector>

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshRendererHeader
	{
		Guid componentId;
		Guid entityId;
		Guid meshId;
		Guid materialIds[8];
		int materialCount;
		bool isStatic;
	};
#pragma pack(pop)

	class MeshRenderer : public Component
	{
		public:
			Guid meshId;
			Guid materialIds[8];
			int materialCount;
			bool isStatic;

		public:
			MeshRenderer();
			MeshRenderer(std::vector<char> data);
			~MeshRenderer();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);
	};

	template <>
	const int ComponentType<MeshRenderer>::type = 3;

	template <typename T>
	struct IsMeshRenderer { static bool value; };

	template <typename T>
	bool IsMeshRenderer<T>::value = false;

	template<>
	bool IsMeshRenderer<MeshRenderer>::value = true;
	template<>
	bool IsComponent<MeshRenderer>::value = true;
}

#endif