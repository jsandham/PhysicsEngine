#include <iostream>
#include <algorithm>
#include <math.h>
#include <cmath>

#include "../../include/core/UniformGrid.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

UniformGrid::UniformGrid()
{

}

UniformGrid::~UniformGrid()
{

}

void UniformGrid::create(Bounds bounds, glm::ivec3 gridDim, std::vector<SphereObject> objects)
{
	if(gridDim.x % 2 != 0 || gridDim.y % 2 != 0 || gridDim.z % 2 != 0){
		std::cout << "Error: Grid dimension for uniform grid must be divisible by 2" << std::endl;
		return;
	}

	this->bounds = bounds;
	this->gridDim = gridDim;
	
	cellSize.x = bounds.size.x / gridDim.x;
	cellSize.y = bounds.size.y / gridDim.y;
	cellSize.z = bounds.size.z / gridDim.z;

	grid.resize(gridDim.x*gridDim.y*gridDim.z, 0);
	count.resize(gridDim.x*gridDim.y*gridDim.z, 0);
	startIndex.resize(gridDim.x*gridDim.y*gridDim.z, -1);

	std::cout << "bounds centre: " << bounds.centre.x << " " << bounds.centre.y << " " << bounds.centre.z << " min: " << bounds.getMin().x << " " << bounds.getMin().y << " " << bounds.getMin().z << " max: " << bounds.getMax().x << " " << bounds.getMax().y << " " << bounds.getMax().z << std::endl;

	firstPass(objects);
	secondPass(objects);

	// create lines array
	lines.resize(6*12*grid.size());

	for(size_t i = 0; i < grid.size(); i++){
		Bounds cellBounds = computeCellBounds((int)i);

		glm::vec3 centre = cellBounds.centre;
		glm::vec3 extent = cellBounds.getExtents();

		// top
		lines[6*12*i] = centre.x - extent.x;
		lines[6*12*i + 1] = centre.y + extent.y;
		lines[6*12*i + 2] = centre.z + extent.z;
		lines[6*12*i + 3] = centre.x + extent.x;
		lines[6*12*i + 4] = centre.y + extent.y;
		lines[6*12*i + 5] = centre.z + extent.z;

		lines[6*12*i + 6] = centre.x + extent.x;
		lines[6*12*i + 7] = centre.y + extent.y;
		lines[6*12*i + 8] = centre.z + extent.z;
		lines[6*12*i + 9] = centre.x + extent.x;
		lines[6*12*i + 10] = centre.y - extent.y;
		lines[6*12*i + 11] = centre.z + extent.z;

		lines[6*12*i + 12] = centre.x + extent.x;
		lines[6*12*i + 13] = centre.y - extent.y;
		lines[6*12*i + 14] = centre.z + extent.z;
		lines[6*12*i + 15] = centre.x - extent.x;
		lines[6*12*i + 16] = centre.y - extent.y;
		lines[6*12*i + 17] = centre.z + extent.z;

		lines[6*12*i + 18] = centre.x - extent.x;
		lines[6*12*i + 19] = centre.y - extent.y;
		lines[6*12*i + 20] = centre.z + extent.z;
		lines[6*12*i + 21] = centre.x - extent.x;
		lines[6*12*i + 22] = centre.y + extent.y;
		lines[6*12*i + 23] = centre.z + extent.z;

		// bottom
		lines[6*12*i + 24] = centre.x - extent.x;
		lines[6*12*i + 25] = centre.y + extent.y;
		lines[6*12*i + 26] = centre.z - extent.z;
		lines[6*12*i + 27] = centre.x + extent.x;
		lines[6*12*i + 28] = centre.y + extent.y;
		lines[6*12*i + 29] = centre.z - extent.z;

		lines[6*12*i + 30] = centre.x + extent.x;
		lines[6*12*i + 31] = centre.y + extent.y;
		lines[6*12*i + 32] = centre.z - extent.z;
		lines[6*12*i + 33] = centre.x + extent.x;
		lines[6*12*i + 34] = centre.y - extent.y;
		lines[6*12*i + 35] = centre.z - extent.z;

		lines[6*12*i + 36] = centre.x + extent.x;
		lines[6*12*i + 37] = centre.y - extent.y;
		lines[6*12*i + 38] = centre.z - extent.z;
		lines[6*12*i + 39] = centre.x - extent.x;
		lines[6*12*i + 40] = centre.y - extent.y;
		lines[6*12*i + 41] = centre.z - extent.z;

		lines[6*12*i + 42] = centre.x - extent.x;
		lines[6*12*i + 43] = centre.y - extent.y;
		lines[6*12*i + 44] = centre.z - extent.z;
		lines[6*12*i + 45] = centre.x - extent.x;
		lines[6*12*i + 46] = centre.y + extent.y;
		lines[6*12*i + 47] = centre.z - extent.z;

		// sides
		lines[6*12*i + 48] = centre.x - extent.x;
		lines[6*12*i + 49] = centre.y + extent.y;
		lines[6*12*i + 50] = centre.z + extent.z;
		lines[6*12*i + 51] = centre.x - extent.x;
		lines[6*12*i + 52] = centre.y + extent.y;
		lines[6*12*i + 53] = centre.z - extent.z;

		lines[6*12*i + 54] = centre.x + extent.x;
		lines[6*12*i + 55] = centre.y + extent.y;
		lines[6*12*i + 56] = centre.z + extent.z;
		lines[6*12*i + 57] = centre.x + extent.x;
		lines[6*12*i + 58] = centre.y + extent.y;
		lines[6*12*i + 59] = centre.z - extent.z;

		lines[6*12*i + 60] = centre.x + extent.x;
		lines[6*12*i + 61] = centre.y - extent.y;
		lines[6*12*i + 62] = centre.z + extent.z;
		lines[6*12*i + 63] = centre.x + extent.x;
		lines[6*12*i + 64] = centre.y - extent.y;
		lines[6*12*i + 65] = centre.z - extent.z;

		lines[6*12*i + 66] = centre.x - extent.x;
		lines[6*12*i + 67] = centre.y - extent.y;
		lines[6*12*i + 68] = centre.z + extent.z;
		lines[6*12*i + 69] = centre.x - extent.x;
		lines[6*12*i + 70] = centre.y - extent.y;
		lines[6*12*i + 71] = centre.z - extent.z;

		if(grid[i] > 0){
			// top
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);

			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);

			// bottom
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			// sides
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y + extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x + extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);

			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z + extent.z);
			occupiedLines.push_back(centre.x - extent.x);
			occupiedLines.push_back(centre.y - extent.y);
			occupiedLines.push_back(centre.z - extent.z);
		}
	}
}

// A Fast Voxel Traversal Algorithm for Ray Tracing
// John Amanatides & Andrew Woo
SphereObject* UniformGrid::intersect(Ray ray)
{
	// find starting voxel
	int cellIndex = computeCellIndex(ray.origin);

	std::cout << "cell Index: " << cellIndex << std::endl;

	if(cellIndex == -1){
		return NULL;
	}

	Bounds cellBounds = computeCellBounds(cellIndex);

	std::cout << "cell index: " << cellIndex << " cell centre: " << cellBounds.centre.x << " " << cellBounds.centre.y << " " << cellBounds.centre.z << std::endl;

	int z = cellIndex / (gridDim.y * gridDim.x);
	cellIndex = cellIndex % (gridDim.y * gridDim.x);
	int y = cellIndex / gridDim.x;
	cellIndex = cellIndex % gridDim.x;
	int x = cellIndex;

	std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

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

		std::cout << "cell index: " << cellIndex << "x: " << x << " y: " << y << " z: " << z << " tMaxX: " << tMaxX << " tMaxY: " << tMaxY << " tMaxZ: " << tMaxZ << std::endl;

		int start = startIndex[cellIndex];
		int end = start + grid[cellIndex];
		if(start != -1){
			for(int i = start; i < end; i++){
				if(Geometry::intersect(ray, sphereObjects[i].sphere)){
					foundIndex = i; // just find first for now
					break;
				}
			}
		}

		if(foundIndex != -1){
			break;
		}
	}

	if(foundIndex >= (int)sphereObjects.size()){
		std::cout << "Error: found index (" << foundIndex << ") is outside array bounds" << std::endl;
		return NULL;
	}

	if(foundIndex != -1){
		std::cout << "found index: " << foundIndex << " sphere id: " << sphereObjects[foundIndex].id.toString() << std::endl;
	}

	return NULL;
}

void UniformGrid::firstPass(std::vector<SphereObject> objects)
{
	for(size_t i = 0; i < objects.size(); i++){
		
		int cellIndex = computeCellIndex(objects[i].sphere.centre);
		if(cellIndex != -1){
			grid[cellIndex]++;

			// std::cout << "i: " << i << " centre: " << objects[i].sphere.centre.x << " " << objects[i].sphere.centre.y << " " << objects[i].sphere.centre.z << " radius: " << objects[i].sphere.radius << " cellIndex: " << cellIndex << " grid: " << grid[cellIndex] << std::endl;

			// check neighbouring 26 cells
			for(int x = -1; x <= 1; x++){
				for(int y = -1; y <= 1; y++){
					for(int z = -1; z <= 1; z++){
						int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
						if(neighbourCellIndex != cellIndex && neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){
							Bounds neighbourCellBounds = computeCellBounds(neighbourCellIndex);
							if(Geometry::intersect(objects[i].sphere, neighbourCellBounds)){
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

	sphereObjects.resize(totalCount);
	for(size_t i = 0; i < sphereObjects.size(); i++){
		sphereObjects[i].sphere.centre = glm::vec3(0.0f, 0.0f, 0.0f);
		sphereObjects[i].sphere.radius = 0.0f;
		sphereObjects[i].id = Guid::INVALID;
	}
}

void UniformGrid::secondPass(std::vector<SphereObject> objects)
{
	for(size_t i = 0; i < objects.size(); i++){

		int cellIndex = computeCellIndex(objects[i].sphere.centre);

		if(cellIndex != -1){
			int location = startIndex[cellIndex] + count[cellIndex];
			count[cellIndex]++;

			sphereObjects[location].sphere = objects[i].sphere;
			sphereObjects[location].id = objects[i].id;

			// check neighbouring 26 cells
			for(int x = -1; x <= 1; x++){
				for(int y = -1; y <= 1; y++){
					for(int z = -1; z <= 1; z++){
						int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
						if(neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){
							Bounds neighbourCellBounds = computeCellBounds(neighbourCellIndex);
							if(Geometry::intersect(objects[i].sphere, neighbourCellBounds)){

								location = startIndex[neighbourCellIndex] + count[neighbourCellIndex];
								count[neighbourCellIndex]++;

								sphereObjects[location].sphere = objects[i].sphere;
								sphereObjects[location].id = objects[i].id;
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
	glm::vec3 localPoint = point - bounds.centre;
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

	glm::vec3 min = bounds.getMin();

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

std::vector<float> UniformGrid::getOccupiedLines() const
{
	return occupiedLines;
}



//    |     |     |     |     |     |     |     |     |
//   -4    -3    -2    -1     0     1     2     3     4