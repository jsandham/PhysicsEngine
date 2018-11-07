#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		public:
			DebugSystem();
			DebugSystem(unsigned char* data);
			~DebugSystem();

			void init();
			void update();
	};
}

#endif