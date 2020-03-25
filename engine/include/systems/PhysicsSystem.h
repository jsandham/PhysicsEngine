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
			std::vector<Collider*> mColliders;
			std::vector<Rigidbody*> mRigidbodies;

			float mTimestep;
			float mGravity;

			bool mStart = false;

		public:
			PhysicsSystem();
			PhysicsSystem(std::vector<char> data);
			~PhysicsSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template <>
	const int SystemType<PhysicsSystem>::type = 1;

	template< typename T>
	struct IsPhysicsSystem { static const bool value; };

	template<typename T>
	const bool IsPhysicsSystem<T>::value = false;

	template<>
	const bool IsPhysicsSystem<PhysicsSystem>::value = true;
	template<>
	const bool IsSystem<PhysicsSystem>::value = true;
}

#endif