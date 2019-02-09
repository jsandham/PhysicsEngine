#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

namespace PhysicsEngine
{
	class CleanUpSystem : public System
	{
		public:
			CleanUpSystem(std::vector<char> data);
			~CleanUpSystem();

			void init();
			void update(Input input);
	};
}

#endif