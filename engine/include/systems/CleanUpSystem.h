#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	class CleanUpSystem : public System
	{
		public:
			CleanUpSystem(Manager *manager, SceneContext* context);
			~CleanUpSystem();

			void init();
			void update();
	};
}

#endif