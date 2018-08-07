#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Rigidbody : public Component
	{
		public:
			bool useGravity;
			float mass;
			float drag;
			float angularDrag;

			glm::vec3 velocity;
			glm::vec3 angularVelocity;
			glm::vec3 centreOfMass;
			glm::mat3 inertiaTensor;

			// leap-frog
			glm::vec3 halfVelocity;

		public:
			Rigidbody();
			~Rigidbody();
	};
}


#endif