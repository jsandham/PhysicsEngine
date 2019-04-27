#include <iostream>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <limits>

#include "../../include/core/UniformGrid.h"
#include "../../include/core/Geometry.h"

#include "../../include/glm/gtx/norm.hpp"

using namespace PhysicsEngine;

UniformGrid::UniformGrid()
{

}

UniformGrid::~UniformGrid()
{

}

void UniformGrid::create(Bounds worldBounds, glm::ivec3 gridDim, std::vector<BoundingSphere> boundingSpheres, std::vector<Sphere> spheres, std::vector<Bounds> bounds, std::vector<Triangle> triangles)
{
	if(gridDim.x % 2 != 0){ gridDim.x--; }
	if(gridDim.y % 2 != 0){ gridDim.y--; }
	if(gridDim.z % 2 != 0){ gridDim.z--; }

	this->worldBounds = worldBounds;
	this->gridDim = gridDim;

	this->boundingSpheres = boundingSpheres;

	this->spheres = spheres;
	this->bounds = bounds;
	this->triangles = triangles;
	
	cellSize.x = worldBounds.size.x / gridDim.x;
	cellSize.y = worldBounds.size.y / gridDim.y;
	cellSize.z = worldBounds.size.z / gridDim.z;

	grid.resize(gridDim.x*gridDim.y*gridDim.z, 0);
	count.resize(gridDim.x*gridDim.y*gridDim.z, 0);
	startIndex.resize(gridDim.x*gridDim.y*gridDim.z, -1);

	std::cout << "world bounds centre: " << worldBounds.centre.x << " " << worldBounds.centre.y << " " << worldBounds.centre.z << " min: " << worldBounds.getMin().x << " " << worldBounds.getMin().y << " " << worldBounds.getMin().z << " max: " << worldBounds.getMax().x << " " << worldBounds.getMax().y << " " << worldBounds.getMax().z << std::endl;

	firstPass(boundingSpheres);
	secondPass(boundingSpheres);

	for(size_t i = 0; i < grid.size(); i++){
		if(grid[i] > 0){
			Bounds cellBounds = computeCellBounds((int)i);

			glm::vec3 centre = cellBounds.centre;
			glm::vec3 extents = cellBounds.getExtents();

			float xf[] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
			float yf[] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
			float zf[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

			// top
			for(int j = 0; j < 8; j++){
				lines.push_back(centre.x + xf[j] * extents.x);
				lines.push_back(centre.y + yf[j] * extents.y);
				lines.push_back(centre.z + zf[j] * extents.z);
			}

			// bottom
			for(int j = 0; j < 8; j++){
				lines.push_back(centre.x + xf[j] * extents.x);
				lines.push_back(centre.y + yf[j] * extents.y);
				lines.push_back(centre.z - zf[j] * extents.z);
			}

			float xg[] = {-1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
			float yg[] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
			float zg[] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

			// sides
			for(int j = 0; j < 8; j++){
				lines.push_back(centre.x + xg[j] * extents.x);
				lines.push_back(centre.y + yg[j] * extents.y);
				lines.push_back(centre.z + zg[j] * extents.z);
			}
		}
	}
}

// A Fast Voxel Traversal Algorithm for Ray Tracing
// John Amanatides & Andrew Woo
BoundingSphere* UniformGrid::intersect(Ray ray)
{
	// find starting voxel
	int cellIndex = computeCellIndex(ray.origin);

	std::cout << "cell Index: " << cellIndex << std::endl;

	if(cellIndex == -1){
		return NULL;
	}

	Bounds cellBounds = computeCellBounds(cellIndex);

	//std::cout << "cell index: " << cellIndex << " cell centre: " << cellBounds.centre.x << " " << cellBounds.centre.y << " " << cellBounds.centre.z << std::endl;

	int z = cellIndex / (gridDim.y * gridDim.x);
	cellIndex = cellIndex % (gridDim.y * gridDim.x);
	int y = cellIndex / gridDim.x;
	cellIndex = cellIndex % gridDim.x;
	int x = cellIndex;

	//std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

	int stepX = ray.direction.x < 0.0f ? -1 : 1;
	int stepY = ray.direction.y < 0.0f ? -1 : 1;
	int stepZ = ray.direction.z < 0.0f ? -1 : 1;

	glm::vec3 cellCentre = cellBounds.centre;
	glm::vec3 cellExtents = cellBounds.getExtents();

	float xmin = cellCentre.x - cellExtents.x;
	float xmax = cellCentre.x + cellExtents.x;
	float ymin = cellCentre.y - cellExtents.y;
	float ymax = cellCentre.y + cellExtents.y;
	float zmin = cellCentre.z - cellExtents.z;
	float zmax = cellCentre.z + cellExtents.z;
	
	float tx0 = (xmin - ray.origin.x) / ray.direction.x;
	float tx1 = (xmax - ray.origin.x) / ray.direction.x;
	float ty0 = (ymin - ray.origin.y) / ray.direction.y;
	float ty1 = (ymax - ray.origin.y) / ray.direction.y;
	float tz0 = (zmin - ray.origin.z) / ray.direction.z;
	float tz1 = (zmax - ray.origin.z) / ray.direction.z;

	float tMaxX = std::max(tx0, tx1);
	float tMaxY = std::max(ty0, ty1);
	float tMaxZ = std::max(tz0, tz1);

	float tDeltaX = abs(tx1 - tx0);
	float tDeltaY = abs(ty1 - ty0);
	float tDeltaZ = abs(tz1 - tz0);

	//std::cout << "tx0: " << tx0 << " ty0: " << ty0 << " tz0: " << tz0 << " tx1: " << tx1 << " ty1: " << ty1 << " tz1: " << tz1 << std::endl;
	//std::cout << "tDeltaX: " << tDeltaX << " tDeltaY: " << tDeltaY << " tDeltaZ: " << tDeltaZ << " tx1: " << tx1 << " ty1: " << ty1 << " tz1: " << tz1 << std::endl;

	int foundIndex = -1;

	while(true){
		if(tMaxX < std::min(tMaxY, tMaxZ)){
			x += stepX;
			if(x < 0 || x >= gridDim.x){
				break;
			}
			tMaxX = tMaxX + tDeltaX;
		}
		else if(tMaxY < std::min(tMaxX, tMaxZ)){
			y += stepY;
			if(y < 0 || y >= gridDim.y){
				break;
			}
			tMaxY = tMaxY + tDeltaY;
		}
		else{
			z += stepZ;
			if(z < 0 || z >= gridDim.z){
				break;
			}
			tMaxZ = tMaxZ + tDeltaZ;
		}

		cellIndex = gridDim.y * gridDim.x * z + gridDim.x * y + x;

		//std::cout << startIndex.size() << std::endl;
		//std::cout << "cell index: " << cellIndex << "x: " << x << " y: " << y << " z: " << z << " tMaxX: " << tMaxX << " tMaxY: " << tMaxY << " tMaxZ: " << tMaxZ << std::endl;

		int start = startIndex[cellIndex];
		int end = start + grid[cellIndex];
		if(start != -1){
			float maxDistanceSqr = std::numeric_limits<float>::max();
			for(int i = start; i < end; i++){
				Sphere sphere = boundingSpheres[data[i]].sphere;

				//std::cout << "primitive type: " << boundingSpheres[data[i]].primitiveType << " data[i]: " << data[i] << " start: " << start << " end: " << end << " checking sphere: " << sphere.centre.x << " " << sphere.centre.y << " " << sphere.centre.z << " radius: " << sphere.radius << std::endl;

				if(Geometry::intersect(ray, sphere)){
					float distanceSqr = glm::length2(ray.origin - sphere.centre) - sphere.radius * sphere.radius;
					if(distanceSqr > 0.0f && distanceSqr < maxDistanceSqr){
						maxDistanceSqr = distanceSqr;
						foundIndex = i;
					}
				}
			}
		}

		if(foundIndex != -1){
			break;
		}
	}

	// if(foundIndex >= (int)sphereObjects.size()){
	if(foundIndex >= (int)data.size()){
		std::cout << "Error: found index (" << foundIndex << ") is outside array world bounds" << std::endl;
		return NULL;
	}

	if(foundIndex != -1){
		// std::cout << "found index: " << foundIndex << " sphere id: " << sphereObjects[foundIndex].id.toString() << std::endl;
		// return &sphereObjects[foundIndex];
		std::cout << "found index: " << foundIndex << " sphere id: " << boundingSpheres[data[foundIndex]].id.toString() << std::endl;
		return &boundingSpheres[data[foundIndex]];
	}

	return NULL;
}

std::vector<BoundingSphere> UniformGrid::intersect(Sphere sphere)
{
	std::vector<BoundingSphere> foundBoundingSpheres;

	// For uniform grids sphere radius cannot be larger than the cell size
	if(sphere.radius > cellSize.x || sphere.radius > cellSize.y || sphere.radius > cellSize.z){
		std::cout << "Warning: Sphere radius must be less than or equal to the uniform grid cell size" << std::endl;
		return foundBoundingSpheres;
	}

	int cellIndex = computeCellIndex(sphere.centre);

	std::cout << "cell index: " << cellIndex << std::endl;

	if(cellIndex != -1){
		for(int x = -1; x <= 1; x++){
			for(int y = -1; y <= 1; y++){
				for(int z = -1; z <= 1; z++){
					int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
					if(neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){

						int start = startIndex[neighbourCellIndex];
						int end = start + grid[neighbourCellIndex];
						for(int i = start; i < end; i++){
							if(Geometry::intersect(boundingSpheres[data[i]].sphere, sphere)){
								foundBoundingSpheres.push_back(boundingSpheres[data[i]]);
							}
						}
					}
				}
			}
		}
	}

	return foundBoundingSpheres;
}

void UniformGrid::firstPass(std::vector<BoundingSphere> boundingSpheres)
{
	for(size_t i = 0; i < boundingSpheres.size(); i++){
		
		int cellIndex = computeCellIndex(boundingSpheres[i].sphere.centre);
		if(cellIndex != -1){
			grid[cellIndex]++;

			// std::cout << "i: " << i << " centre: " << boundingSpheres[i].sphere.centre.x << " " << boundingSpheres[i].sphere.centre.y << " " << boundingSpheres[i].sphere.centre.z << " radius: " << boundingSpheres[i].sphere.radius << " cellIndex: " << cellIndex << " grid: " << grid[cellIndex] << std::endl;

			// check neighbouring 26 cells
			for(int x = -1; x <= 1; x++){
				for(int y = -1; y <= 1; y++){
					for(int z = -1; z <= 1; z++){
						int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
						if(neighbourCellIndex != cellIndex && neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){
							Bounds neighbourCellBounds = computeCellBounds(neighbourCellIndex);
							if(Geometry::intersect(boundingSpheres[i].sphere, neighbourCellBounds)){
								grid[neighbourCellIndex]++;
							}
						}
					}
				}
			}
		}
	}

	int totalCount = 0;
	for(size_t i = 0; i < grid.size(); i++){
		if(grid[i] > 0){
			startIndex[i] = totalCount;
			totalCount += grid[i];
		}
	}

	std::cout << "total count: " << totalCount << std::endl;

	data.resize(totalCount);
	for(size_t i = 0; i < data.size(); i++){
		data[i] = -1;
	}
}

void UniformGrid::secondPass(std::vector<BoundingSphere> boundingSpheres)
{
	for(size_t i = 0; i < boundingSpheres.size(); i++){

		int cellIndex = computeCellIndex(boundingSpheres[i].sphere.centre);

		if(cellIndex != -1){
			int location = startIndex[cellIndex] + count[cellIndex];
			count[cellIndex]++;

			// sphereObjects[location].sphere = boundingSpheres[i].sphere;
			// sphereObjects[location].id = boundingSpheres[i].id;
			data[location] = (int)i;//boundingSpheres[i].index;

			// check neighbouring 26 cells
			for(int x = -1; x <= 1; x++){
				for(int y = -1; y <= 1; y++){
					for(int z = -1; z <= 1; z++){
						int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
						if(neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){  //////////////////// should I add neighbourCellIndex != cellIndex check here????
							Bounds neighbourCellBounds = computeCellBounds(neighbourCellIndex);
							if(Geometry::intersect(boundingSpheres[i].sphere, neighbourCellBounds)){

								location = startIndex[neighbourCellIndex] + count[neighbourCellIndex];
								count[neighbourCellIndex]++;

								// sphereObjects[location].sphere = boundingSpheres[i].sphere;
								// sphereObjects[location].id = boundingSpheres[i].id;
								data[location] = (int)i;//boundingSpheres[i].index;
							}
						}
					}
				}
			}
		}
	}
}

int UniformGrid::computeCellIndex(glm::vec3 point) const
{
	glm::vec3 localPoint = point - worldBounds.centre;
	glm::ivec3 gridPos = glm::ivec3(floor(localPoint.x / cellSize.x), floor(localPoint.y / cellSize.y), floor(localPoint.z / cellSize.z));
	gridPos += gridDim / 2;

	if(gridPos.x < 0 || gridPos.x >= gridDim.x || gridPos.y < 0 || gridPos.y >= gridDim.y || gridPos.z < 0 || gridPos.z >= gridDim.z){
		return -1;
	}
	else{
		return gridDim.y * gridDim.x * gridPos.z + gridDim.x * gridPos.y + gridPos.x;
	}
}

Bounds UniformGrid::computeCellBounds(int cellIndex) const
{
	glm::ivec3 gridPos = glm::ivec3(0, 0, 0);

	int index = cellIndex;
	gridPos.z = index / (gridDim.x * gridDim.y);
	index = index % (gridDim.x * gridDim.y);
	gridPos.y = index / gridDim.x;
	index = index % (gridDim.x);
	gridPos.x = index;

	glm::vec3 min = worldBounds.getMin();

	Bounds cellBounds;
	cellBounds.size = cellSize;
	cellBounds.centre.x = (gridPos.x + 0.5f) * cellSize.x + min.x;
	cellBounds.centre.y = (gridPos.y + 0.5f) * cellSize.y + min.y;
	cellBounds.centre.z = (gridPos.z + 0.5f) * cellSize.z + min.z;

	return cellBounds;
}

std::vector<float> UniformGrid::getLines() const
{
	return lines;
}



//    |     |     |     |     |     |     |     |     |
//   -4    -3    -2    -1     0     1     2     3     4