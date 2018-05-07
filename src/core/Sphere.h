#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Sphere
	{
		public:
			float radius;
			glm::vec3 centre;

		public:
			Sphere();
			Sphere(glm::vec3 centre, float radius);
			~Sphere();
	};
}
#endif