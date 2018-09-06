#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <map>

#include "../core/SceneContext.h"
#include "../core/Manager.h"

namespace PhysicsEngine
{
	class System
	{
		protected:
			SceneContext* context;
			Manager* manager;

		public:
			System();
			virtual ~System() = 0;

			virtual void init() = 0;
			virtual void update() = 0;
	};
}

#endif