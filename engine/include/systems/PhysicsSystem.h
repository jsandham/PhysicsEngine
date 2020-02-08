#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../components/Collider.h"
#include "../components/Rigidbody.h"

namespace PhysicsEngine
{
	class PhysicsSystem : public System
	{
		private:
			std::vector<Collider*> colliders;
			std::vector<Rigidbody*> rigidbodies;

			float timestep;
			float gravity;

			bool start = false;

		public:
			PhysicsSystem();
			PhysicsSystem(std::vector<char> data);
			~PhysicsSystem();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template< typename T>
	struct IsPhysicsSystem { static bool value; };

	template<typename T>
	bool IsPhysicsSystem<T>::value = false;

	template<>
	bool IsPhysicsSystem<PhysicsSystem>::value = true;
	template<>
	bool IsSystem<PhysicsSystem>::value = true;
}

#endif