#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include "../core/SceneContext.h"

namespace PhysicsEngine
{
	class Manager;

	class System
	{
		protected:
			SceneContext* context;
			Manager* manager;

		public:
			System();
			virtual ~System() = 0;

			virtual size_t getSize() = 0;
			virtual void init() = 0;
			virtual void update() = 0;

			void setManager(Manager* manager);
			void setSceneContext(SceneContext* context);
	};

	System* loadSystem(unsigned char* data);
}

#endif