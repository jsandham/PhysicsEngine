#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct RigidbodyHeader
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
#pragma pack(pop)

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
			Rigidbody(std::vector<char> data);
			~Rigidbody();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);
	};

	template <>
	const int ComponentType<Rigidbody>::type = 1;

	template <typename T>
	struct IsRigidbody { static bool value; };

	template <typename T>
	bool IsRigidbody<T>::value = false;

	template<>
	bool IsRigidbody<Rigidbody>::value = true;
	template<>
	bool IsComponent<Rigidbody>::value = true;
}


#endif