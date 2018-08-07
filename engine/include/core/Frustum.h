#ifndef __FRUSTUM_H__
#define __FRUSTUM_H__

#undef NEAR
#undef FAR
#undef near
#undef far

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	// plane defined by n.x*x + n.y*y + n.z*z + d = 0, where d = -dot(n, x0)
	class Plane
	{
		public:
			glm::vec3 n;
			glm::vec3 x0;

			float distance(glm::vec3 point);
	};

	class Frustum
	{
		private:
			Plane planes[6];

			enum {
				TOP = 0,
				BOTTOM,
				LEFT,
				RIGHT,
				NEAR,
				FAR
			};

			// update when perspective changes
			float angle;
			float ratio;
			float near;
			float far;

			float nearPlaneHeight;
			float nearPlaneWidth;
			float farPlaneHeight;
			float farPlaneWidth;

			// update when camera changes
			glm::vec3 position;
			glm::vec3 front;
			glm::vec3 up;
			glm::vec3 right;

			// vertices and normals for drawing the frustum box
			glm::vec3 frustumVertices[36];
			glm::vec3 frustumNormals[36];

		public:
			Frustum();
			~Frustum();

			void setPerspective(float angle, float ratio, float near, float far);
			void setCamera(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right);
			void setCameraSlow(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right);

			int checkPoint(glm::vec3 point);
			int checkSphere(glm::vec3 centre, float radius);
			int checkAABB(glm::vec3 min, glm::vec3 max);
	};

}
#endif