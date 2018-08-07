#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <map>

#include "../core/Entity.h"
#include "../core/Manager.h"

namespace PhysicsEngine
{
	class System
	{
		protected:
			Manager *manager;

		public:
			System();
			virtual ~System() = 0;

			virtual void init() = 0;
			virtual void update() = 0;
	};
}

#endif