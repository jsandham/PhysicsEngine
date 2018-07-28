#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <vector>

namespace PhysicsEngine
{
	class Entity;

	typedef enum ComponentType
	{
		TransformType,
		RigidbodyType,
		MeshRendererType,
		DirectionalLightType,
		SpotLightType,
		PointLightType	
	};

	class Component
	{
		public:
			int globalEntityIndex;
			int globalComponentIndex;
			int componentId;
			int entityId;

		public:
			Component();
			virtual ~Component() = 0;

			Entity* Component::getEntity(std::vector<Entity*> entities);
	};
}

#endif