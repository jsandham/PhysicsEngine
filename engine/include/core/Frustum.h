#ifndef __FRUSTUM_H__
#define __FRUSTUM_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "Plane.h"

namespace PhysicsEngine
{
	class Frustum
	{
		public:
			// normals in frustum plane are assumed to point inward
			Plane mPlanes[6];

			float mFov;          // vertical fov
			float mAspectRatio;  // determines horizontal fov
			float mNearPlane;
			float mFarPlane;

			enum {
				TOP = 0,
				BOTTOM,
				LEFT,
				RIGHT,
				NEAR,
				FAR
			};

		public:
			Frustum();
			~Frustum();

			void computePlanes(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right);
			bool containsPoint(glm::vec3 point) const;
	};
}

#endif