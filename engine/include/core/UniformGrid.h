#ifndef __UNIFORMGRID_H__
#define __UNIFORMGRID_H__	

#include <vector>

#include "../glm/glm.hpp"

#include "Guid.h"
#include "Ray.h"
#include "Sphere.h"
#include "Bounds.h"
#include "Triangle.h"

namespace PhysicsEngine
{
	typedef struct BoundingSphere
	{
		Guid id;
		Sphere sphere;
		int primitiveType;
		int index;
	}BoundingSphere;


	// UniformGrid only works with spheres, Prob use bounding spheres of other collider types
	class UniformGrid // rename to SUniformGrid? or StaticUniformGrid?
	{
		private:
			Bounds worldBounds;
			glm::ivec3 gridDim;
			glm::vec3 cellSize;
			std::vector<int> grid;
			std::vector<int> count;
			std::vector<int> startIndex;
			std::vector<int> data;

			std::vector<BoundingSphere> boundingSpheres;
			std::vector<Sphere> spheres;
			std::vector<Bounds> bounds;
			std::vector<Triangle> triangles;

			std::vector<float> lines;
			//std::vector<float> occupiedLines;

		public:
			UniformGrid();
			~UniformGrid();

			void create(Bounds worldBounds, glm::ivec3 gridDim, std::vector<BoundingSphere> boundingSpheres, std::vector<Sphere> spheres, std::vector<Bounds> bounds, std::vector<Triangle> triangles);

			BoundingSphere* intersect(Ray ray);
			std::vector<BoundingSphere> intersect(Sphere sphere);
			std::vector<float> getLines() const;

		private:
			void firstPass(std::vector<BoundingSphere> boundingSpheres);
			void secondPass(std::vector<BoundingSphere> boundingSpheres);
			int computeCellIndex(glm::vec3 point) const;
			Bounds computeCellBounds(int cellIndex) const;
	};
}


#endif