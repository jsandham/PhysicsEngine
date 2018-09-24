#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	class CleanUpSystem : public System
	{
		public:
			// CleanUpSystem(Manager *manager, SceneContext* context);
			CleanUpSystem();
			~CleanUpSystem();

			size_t getSize();
			void init();
			void update();
	};
}

#endif