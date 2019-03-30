#ifndef __UNIFORMGRID_H__
#define __UNIFORMGRID_H__	

#include <vector>

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	// UniformGrid only works with spheres, Prob use bounding spheres of other collider types
	class UniformGrid // rename to SUniformGrid? or StaticUniformGrid?
	{
		private:
			glm::ivec3 gridSize;
			int hashTableSize;

			std::vector<int> hashTable; 
			std::vector<Object> objects;

		public:
			UniformGrid();
			~UniformGrid();

			void create(glm::vec3 gridSize, int hashTableSize);
			void firstPass(std::vector<Sphere> spheres);
			void secondPass(std::vector<Sphere>, std::vector<Guid> ids);

			Object* intersect(Ray ray);

		private:
			// int computeHash(glm::vec3 centre);
			int computeHash(glm::ivec3 centre);
	};
}


#endif