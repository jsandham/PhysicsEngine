#include <iostream>
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
	std::cout << "Uniform grid create called" << std::endl;

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
	startIndex.resize(gridDim.x*gridDim.y*gridDim.z, -1);

	sphereObjects.resize(objects.size());
	for(size_t i = 0; i < sphereObjects.size(); i++){
		sphereObjects[i].sphere.centre = glm::vec3(0.0f, 0.0f, 0.0f);
		sphereObjects[i].sphere.radius = 0.0f;
		sphereObjects[i].id = Guid::INVALID;
	}

	firstPass(objects);
	//secondPass(objects);

	std::cout << "bounds centre: " << bounds.centre.x << " " << bounds.centre.y << " " << bounds.centre.z << " min: " << bounds.getMin().x << " " << bounds.getMin().y << " " << bounds.getMin().z << " max: " << bounds.getMax().x << " " << bounds.getMax().y << " " << bounds.getMax().z << std::endl;

	Bounds test = computeCellBounds(0);
	std::cout << "Grid size: " << grid.size() << " " << test.centre.x << " " << test.centre.y << " " << test.centre.z << std::endl;

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
	// bool flag = true;
	// while(flag);  
	// { 
	// 	if(tMaxX < tMaxY) { 
	// 		if(tMaxX < tMaxZ) { 
	// 			X= X + stepX; 
	// 			if(X == justOutX) 
	// 				return(NIL); /* outside grid */ 
	// 			tMaxX= tMaxX + tDeltaX; 
	// 		} 
	// 		else { 
	// 			Z= Z + stepZ; 
	// 			if(Z == justOutZ) 
	// 				return(NIL); 
	// 			tMaxZ= tMaxZ + tDeltaZ; 
	// 		} 
	// 	} 
	// 	else { 
	// 		if(tMaxY < tMaxZ) { 
	// 			Y= Y + stepY; 
	// 			if(Y == justOutY) 
	// 				return(NIL); 
	// 			tMaxY= tMaxY + tDeltaY; 
	// 		} 
	// 		else { 
	// 			Z= Z + stepZ; 
	// 			if(Z == justOutZ) 
	// 				return(NIL); 
	// 			tMaxZ= tMaxZ + tDeltaZ; 
	// 		} 
	// 	} 
	// 	list= ObjectList[X][Y][Z]; 
	// } 

	// return(list);


	return NULL;
}

void UniformGrid::firstPass(std::vector<SphereObject> objects)
{
	for(size_t i = 0; i < objects.size(); i++){
		
		int cellIndex = computeCellIndex(objects[i].sphere.centre);
		if(cellIndex != -1){
			grid[cellIndex]++;

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
}

void UniformGrid::secondPass(std::vector<SphereObject> objects)
{
	for(size_t i = 0; i < objects.size(); i++){
		int cellIndex = computeCellIndex(objects[i].sphere.centre);
		if(cellIndex != -1){
			int start = startIndex[cellIndex];
			int end = start + grid[cellIndex];

			if(start == -1){
				std::cout << "Error: start index of -1 found during second pass of uniform grid generation" << std::endl;
				return;
			}

			for(int j = start; j < end; j++){
				if(sphereObjects[j].id == Guid::INVALID){
					sphereObjects[j].sphere = objects[i].sphere;
					sphereObjects[j].id = objects[i].id;
					break;
				}
			}

			// check neighbouring 26 cells
			for(int x = -1; x <= 1; x++){
				for(int y = -1; y <= 1; y++){
					for(int z = -1; z <= 1; z++){
						int neighbourCellIndex = cellIndex + gridDim.x * gridDim.y * z + gridDim.x * y + x;
						if(neighbourCellIndex != cellIndex && neighbourCellIndex >= 0 && neighbourCellIndex < grid.size()){
							Bounds cellBounds = computeCellBounds(neighbourCellIndex);
							if(Geometry::intersect(objects[i].sphere, cellBounds)){
								start = startIndex[neighbourCellIndex];
								end = start + grid[neighbourCellIndex];

								if(start == -1){
									std::cout << "Error: start index of -1 found during second pass of uniform grid generation" << std::endl;
									return;
								}

								for(int j = start; j < end; j++){
									if(sphereObjects[j].id == Guid::INVALID){
										sphereObjects[j].sphere = objects[i].sphere;
										sphereObjects[j].id = objects[i].id;
										break;
									}
								}
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


