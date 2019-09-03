#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <vector>

#include "../core/Input.h"

namespace PhysicsEngine
{
	class World;

	class System
	{
		protected:
			int order;

			World* world;

		public:
			System();
			virtual ~System() = 0;

			virtual std::vector<char> serialize() = 0;
			virtual void deserialize(std::vector<char> data) = 0;

			virtual void init(World* world) = 0;
			virtual void update(Input input) = 0;

			int getOrder() const;
	};
}

#endif