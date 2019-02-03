#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct RigidbodyData
	{
		Guid componentId;
		Guid entityId;
		bool useGravity;
		float mass;
		float drag;
		float angularDrag;

		glm::vec3 velocity;
		glm::vec3 angularVelocity;
		glm::vec3 centreOfMass;
		glm::mat3 inertiaTensor;
		
		glm::vec3 halfVelocity;
	};
// #pragma pack(pop)

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
			Rigidbody(unsigned char* data);
			~Rigidbody();

			void load(RigidbodyData data);
	};
}


#endif