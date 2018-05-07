#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include "System.h"

#include "../graphics/Material.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		public:
			DebugSystem(Manager* manager);
			~DebugSystem();

			void init();
			void update();
	};
}

#endif