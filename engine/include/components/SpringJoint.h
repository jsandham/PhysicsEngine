#ifndef __SPRING_JOINT_H__
#define __SPRING_JOINT_H__

#include "Joint.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class SpringJoint : public Joint
	{
		public:
			float damping;
			float stiffness;
			float restLength;
			float minDistance;
			float maxDistance;

		public:
			SpringJoint();
			~SpringJoint();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			glm::vec3 getTargetPosition();
	};
}

#endif