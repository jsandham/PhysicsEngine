#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Rigidbody : public Component
	{
		private:
			glm::vec3 angularVelocity;
			glm::vec3 centreOfMass;
			glm::mat3 inertiaTensor;

		public:
			float mass;
			float gravity;
			float drag;
			float angularDrag;

		public:
			Rigidbody();
			Rigidbody(Entity *entity);
			~Rigidbody();

			glm::vec3 GetAngularVelocity();
			glm::vec3 GetCentreOfMass();
			glm::mat3 GetInertiaTensor();
	};
}


#endif