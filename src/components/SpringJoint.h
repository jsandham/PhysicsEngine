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
			SpringJoint(Entity* entity);
			~SpringJoint();

			glm::vec3 getTargetPosition();
	};
}

#endif