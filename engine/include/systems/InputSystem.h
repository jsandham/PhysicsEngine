#ifndef __INPUTSYSTEM_H__
#define __INPUTSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	class InputSystem : public System
	{
		public:
			InputSystem();
			~InputSystem();

			void init(World* world);
			void update(Input input);
	};
}

#endif