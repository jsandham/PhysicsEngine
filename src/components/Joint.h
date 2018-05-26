#ifndef __JOINT_H__
#define __JOINT_H__

#include "Component.h"
#include "Rigidbody.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Joint : public Component
	{
		protected:
			Joint* connectedJoint;
			Rigidbody* connectedBody;
			glm::vec3 connectedAnchor;
			glm::vec3 anchor;

		public:
			Joint();
			virtual ~Joint() = 0;

			Joint* getConnectedJoint();
			Rigidbody* getConnectedBody();
			glm::vec3 getConnectedAnchor();
			glm::vec3 getAnchor();
			
			void setConnectedJoint(Joint* joint);
			void setConnectedBody(Rigidbody* body);
			void setConnectedAnchor(glm::vec3 anchor);
			void setAnchor(glm::vec3 anchor);
	};
}

#endif