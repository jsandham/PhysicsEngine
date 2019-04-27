#ifndef __HGRID_H__
#define __HGRID_H__

#include <vector>

#include "../glm/glm.hpp"

#include "Guid.h"
#include "Ray.h"
#include "Sphere.h"
#include "Bounds.h"
#include "UniformGrid.h"

namespace PhysicsEngine
{
	typedef struct HGridLevel
	{
		glm::vec3 cellSize;
		std::vector<int> grid;
		std::vector<int> count;
		std::vector<int> startIndex;
		std::vector<BoundingSphere> boundingSpheres;
	}HGridLevel;

	class HGrid
	{
		private:
			Bounds bounds;
			std::vector<HGridLevel> levels;

		public:
			HGrid();
			~HGrid();

			void create(Bounds bounds, glm::ivec3 gridDim, std::vector<BoundingSphere> boundingSpheres);

			BoundingSphere* intersect(Ray ray);
			std::vector<float> getLines() const;
			std::vector<float> getOccupiedLines() const;

		private:
			int computeCellIndex(glm::vec3 point) const;
			Bounds computeCellBounds(int cellIndex) const;
			
	};
}

#endif