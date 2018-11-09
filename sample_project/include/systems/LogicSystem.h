#ifndef __LOGICSYSTEM_H__
#define __LOGICSYSTEM_H__

#include <systems/System.h>

namespace PhysicsEngine
{
	class LogicSystem : public System
	{
		public:
			LogicSystem();
			LogicSystem(unsigned char* data);
			~LogicSystem();

			void init();
			void update();
	};
}

#endif