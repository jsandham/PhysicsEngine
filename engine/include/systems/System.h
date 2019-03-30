#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include "../core/Input.h"

namespace PhysicsEngine
{
	class World;

	class System
	{
		protected:
			int type;

			World* world;

		public:
			System();
			virtual ~System() = 0;

			virtual void init(World* world) = 0;
			virtual void update(Input input) = 0;
	};
}

#endif