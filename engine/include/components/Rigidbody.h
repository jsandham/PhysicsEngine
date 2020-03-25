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
		Guid mComponentId;
		Guid mEntityId;
		bool mUseGravity;
		float mMass;
		float mDrag;
		float mAngularDrag;

		glm::vec3 mVelocity;
		glm::vec3 mAngularVelocity;
		glm::vec3 mCentreOfMass;
		glm::mat3 mInertiaTensor;
		
		glm::vec3 mHalfVelocity;
	};
#pragma pack(pop)

	class Rigidbody : public Component
	{
		public:
			bool mUseGravity;
			float mMass;
			float mDrag;
			float mAngularDrag;

			glm::vec3 mVelocity;
			glm::vec3 mAngularVelocity;
			glm::vec3 mCentreOfMass;
			glm::mat3 mInertiaTensor;

			// leap-frog
			glm::vec3 mHalfVelocity;

		public:
			Rigidbody();
			Rigidbody(std::vector<char> data);
			~Rigidbody();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);
	};

	template <>
	const int ComponentType<Rigidbody>::type = 1;

	template <typename T>
	struct IsRigidbody { static const bool value; };

	template <typename T>
	const bool IsRigidbody<T>::value = false;

	template<>
	const bool IsRigidbody<Rigidbody>::value = true;
	template<>
	const bool IsComponent<Rigidbody>::value = true;
}


#endif