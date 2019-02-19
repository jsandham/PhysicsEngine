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

		void* operator new(size_t size);
		void operator delete(void*);

		void init(World* world);
		void update(Input input);
	};
}

#endif