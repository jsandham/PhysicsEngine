#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <map>

#include "../entities/Entity.h"
#include "../core/Log.h"

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