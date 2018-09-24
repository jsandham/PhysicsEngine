#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		public:
			// DebugSystem(Manager* manager, SceneContext* context);
			DebugSystem();
			~DebugSystem();

			size_t getSize();
			void init();
			void update();
	};
}

#endif