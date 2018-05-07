#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <map>

namespace PhysicsEngine
{
	class Entity;

	class Component
	{
		public:
			Entity *entity;

		public:
			Component();
			virtual ~Component() = 0;
	};
}

#endif