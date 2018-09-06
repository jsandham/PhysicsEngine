#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <vector>

namespace PhysicsEngine
{
	class Entity;
	class Manager;

	typedef enum ComponentType
	{
		TransformType,
		RigidbodyType,
		CameraType,
		MeshRendererType,
		DirectionalLightType,
		SpotLightType,
		PointLightType	
	};

	class Component
	{
		private:
			Manager* manager;

		public:
			bool isActive;

			int globalEntityIndex;
			int globalComponentIndex;
			int componentId;
			int entityId;

		public:
			Component();
			virtual ~Component() = 0;

			void setManager(Manager* manager);

			Entity* getEntity();

			template<typename T>
			T* getComponent()
			{
				Entity* entity = getEntity();

				return entity->getComponent<T>();
			}
	};
}

#endif