#ifndef __LOGICSYSTEM_H__
#define __LOGICSYSTEM_H__

#include <vector>

#include <systems/System.h>

#include <core/Input.h>

namespace PhysicsEngine
{
	class LogicSystem : public System
	{
	public:
		LogicSystem();
		LogicSystem(std::vector<char> data);
		~LogicSystem();

		std::vector<char> serialize();
		void deserialize(std::vector<char> data);

		void init(World* world);
		void update(Input input);
	};
}

#endif