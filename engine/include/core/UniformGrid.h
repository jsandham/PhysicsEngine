#ifndef __UNIFORMGRID_H__
#define __UNIFORMGRID_H__	

#include <vector>

#include "../glm/glm.hpp"

#include "../../include/core/Guid.h"
#include "../../include/core/Ray.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Bounds.h"

namespace PhysicsEngine
{
	// uniform grids only operate of sphere objects.
	typedef struct SphereObject
	{
		Guid id;
		Sphere sphere; 
	}SphereObject;


	// UniformGrid only works with spheres, Prob use bounding spheres of other collider types
	class UniformGrid // rename to SUniformGrid? or StaticUniformGrid?
	{
		private:
			Bounds bounds;
			glm::ivec3 gridDim;
			glm::vec3 cellSize;
			std::vector<int> grid;
			std::vector<int> startIndex;
			std::vector<SphereObject> sphereObjects;

			std::vector<int> count;

			std::vector<float> lines;
			std::vector<float> occupiedLines;

		public:
			UniformGrid();
			~UniformGrid();

			void create(Bounds bounds, glm::ivec3 gridDim, std::vector<SphereObject> objects);

			SphereObject* intersect(Ray ray);
			std::vector<float> getLines() const;
			std::vector<float> getOccupiedLines() const;

		private:
			void firstPass(std::vector<SphereObject> objects);
			void secondPass(std::vector<SphereObject> objects);
			int computeCellIndex(glm::vec3 point) const;
			Bounds computeCellBounds(int cellIndex) const;
	};
}


#endif