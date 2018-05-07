#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include "System.h"

#include "../memory/Manager.h"

namespace PhysicsEngine
{
	class CleanUpSystem : public System
	{
		public:
			CleanUpSystem(Manager *manager);
			~CleanUpSystem();

			void init();
			void update();
	};
}

#endif