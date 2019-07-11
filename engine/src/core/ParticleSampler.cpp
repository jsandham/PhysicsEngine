#include "../../include/core/ParticleSampler.h"

#include <random>
#include <stack>
#include <list>
#include <algorithm>
#include <math.h>
#include <iostream>

#define PI 3.14159265f

using namespace PhysicsEngine;

void ParticleSampler::poissonSampler(std::vector<float> &points, float minx, float miny, float minz, float maxx, float maxy, float maxz, float h, float r, unsigned int k)
{
	std::vector<int> grid;
	std::vector<float> newPoints;

	float dr = r / sqrtf(3);

	int nx = (int)ceil((maxx - minx) / dr);
	int ny = (int)ceil((maxy - miny) / dr);
	int nz = (int)ceil((maxz - minz) / dr);

	grid.resize(nx * ny * nz);

	for (unsigned int i = 0; i < grid.size(); i++){
		grid[i] = -1;
	}

	// random number generators
	std::default_random_engine generator;
	std::uniform_real_distribution<float> dist_x(minx, maxx);
	std::uniform_real_distribution<float> dist_y(miny, maxy);
	std::uniform_real_distribution<float> dist_z(minz, maxz);
	std::uniform_real_distribution<float> radius_dist(r, 2 * r);
	std::uniform_real_distribution<float> theta_dist(0, PI);
	std::uniform_real_distribution<float> phi_dist(0, 2 * PI);

	//std::stack<int> activeStack;
	std::vector<int> active;

	// select initial point
	float x_0 = dist_x(generator);
	float y_0 = dist_y(generator);
	float z_0 = dist_z(generator);
	//glm::vec3 x_0 = glm::vec3(dist_x(generator), dist_y(generator), dist_z(generator));
	newPoints.push_back(x_0);
	newPoints.push_back(y_0);
	newPoints.push_back(z_0);
	//activeStack.push(0);
	active.push_back(0);
	int index = findGridLocation(x_0, y_0, z_0, minx, miny, minz, nx, ny, dr);
	grid[index] = 0;

	int count = 0;
	//while (!activeStack.empty()){
	while (!active.empty()){
		count++;

		// pop top index off the active stack
		//int i = activeStack.top();
		std::uniform_int_distribution<int> random(0, (int)active.size() - 1);
		int rand = random(generator);
		int i = active[rand];

		// find the point x_i
		float x_i = newPoints[3*i];
		float y_i = newPoints[3*i + 1];
		float z_i = newPoints[3*i + 2];
		//glm::vec3 x_i = newPoints[i];

		// generate up to k points choosen uniformly from the spherical annulus between radius r and 2r around x_i
		bool sampleFound = false;
		for (unsigned int j = 0; j < k; j++){
			float radius = radius_dist(generator);
			float theta = theta_dist(generator);
			float phi = phi_dist(generator);

			float x = x_i + radius*sin(theta)*cos(phi);
			float y = y_i + radius*sin(theta)*sin(phi);
			float z = z_i + radius*cos(theta);

			if (x >= minx && x <= maxx && y >= miny && y <= maxy && z >= minz && z <= maxz){
				float k_pointx = x;
				float k_pointy = y;
				float k_pointz = z;
				//glm::vec3 k_point = glm::vec3(x, y, z);

				// for each point in turn, check if it is within distance r of existing points
				index = findGridLocation(k_pointx, k_pointy, k_pointz, minx, miny, minz, nx, ny, dr);

				if (index >= nx*ny*nz){
					std::cout << "index: " << index << std::endl;
				}

				if (grid[index] != -1){
					continue;
				}

				if (!IsNearAnotherPoint(grid, newPoints, k_pointx, k_pointy, k_pointz, index, nx, ny, nz, r)){
					sampleFound = true;
					newPoints.push_back(k_pointx);
					newPoints.push_back(k_pointy);
					newPoints.push_back(k_pointz);
					grid[index] = (int)newPoints.size()/3 - 1;
					active.push_back((int)newPoints.size()/3 - 1);
					//activeStack.push((int)newPoints.size() - 1);
				}
			}
		}

		if (sampleFound == false){
			// swap 
			active[rand] = active[active.size() - 1];
			active.resize(active.size() - 1);
			//activeStack.pop();
		}

		//if (i % 100 == 0){
		//	//std::cout << "count: " << count << " stack size: " << activeStack.size() << " points.size(): " << newPoints.size() << " grid.size(): " << grid.size() << std::endl;
		//	std::cout << "count: " << count << " stack size: " << active.size() << " points.size(): " << newPoints.size() << " grid.size(): " << grid.size() << std::endl;
		//}
	}

	for (unsigned int j = 0; j < newPoints.size(); j++){
		points.push_back(newPoints[j]);
	}
}

void ParticleSampler::randomParticles(std::vector<float> &points, float minx, float miny, float minz, float maxx, float maxy, float maxz, float h, int numPoints)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distributionx(minx + h, maxx - h);
	std::uniform_real_distribution<float> distributiony(miny + h, maxy - h);
	std::uniform_real_distribution<float> distributionz(minz + h, maxz - h);

	points.resize(3*numPoints);

	for (int i = 0; i < numPoints; i++){
		points[3*i] = 1.0f*distributionx(generator);
		points[3*i + 1] = 0.25f*distributiony(generator);
		points[3*i + 2] = 0.5f*distributionz(generator);
	}
}

void ParticleSampler::gridOfParticlesXYPlane(std::vector<float> &points, float dx, float dy, float z, int nx, int ny)
{
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < ny; j++){
			points.push_back(i*dx);
			points.push_back(j*dy);
			points.push_back(z);
		}
	}
}

void ParticleSampler::gridOfParticlesXZPlane(std::vector<float> &points, float dx, float dz, float y, int nx, int nz)
{
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < nz; j++){
			points.push_back(i*dx);
			points.push_back(y);
			points.push_back(j*dz);
		}
	}
}


int ParticleSampler::findGridLocation(float x, float y, float z, float minx, float miny, float minz, int nx, int ny, float dr)
{
	return ny * nx * (int)floor((z - minz) / dr) + nx * (int)floor((y - miny) / dr) + (int)floor((x - minx) / dr);
}

bool ParticleSampler::IsNearAnotherPoint(std::vector<int> &grid, std::vector<float> &points, float x, float y, float z, int index, int nx, int ny, int nz, float r)
{
	for (int a = -2; a <= 2; a++){
		for (int b = -2; b <= 2; b++){
			for (int c = -2; c <= 2; c++){
				int index2 = index + nx*ny*c + nx*b + a;
				if (index2 >= 0 && index2 < nx*ny*nz){
					if (grid[index2] != -1){
						float otherPointx = points[3*grid[index2]];
						float otherPointy = points[3*grid[index2] + 1];
						float otherPointz = points[3*grid[index2] + 2];
						float distance = (otherPointx - x)*(otherPointx - x) + (otherPointy - y)*(otherPointy - y) + (otherPointz - z)*(otherPointz - z);

						if (distance < r*r){
							return true;
						}
					}
				}
			}
		}
	}

	return false;
}




























//void ParticleSampler::poissonSampler(std::vector<glm::vec3> &points, glm::vec3 min, glm::vec3 max, float h, float r, unsigned int k)
//{
//	std::vector<int> grid;
//	std::vector<glm::vec3> newPoints;
//
//	float dr = r / sqrtf(3);
//	
//	int nx = (int)ceil((max.x - min.x) / dr);
//	int ny = (int)ceil((max.y - min.y) / dr);
//	int nz = (int)ceil((max.z - min.z) / dr);
//
//	grid.resize(nx * ny * nz);
//
//	for (unsigned int i = 0; i < grid.size(); i++){
//		grid[i] = -1;
//	}
//
//	// random number generators
//	std::default_random_engine generator;
//	std::uniform_real_distribution<float> dist_x(min.x, max.x);
//	std::uniform_real_distribution<float> dist_y(min.y, max.y);
//	std::uniform_real_distribution<float> dist_z(min.z, max.z);
//	std::uniform_real_distribution<float> radius_dist(r, 2 * r);
//	std::uniform_real_distribution<float> theta_dist(0, PI);
//	std::uniform_real_distribution<float> phi_dist(0, 2 * PI);
//
//	//std::stack<int> activeStack;
//	std::vector<int> active;
//	
//	// select initial point
//	glm::vec3 x_0 = glm::vec3(dist_x(generator), dist_y(generator), dist_z(generator));
//	newPoints.push_back(x_0);
//	//activeStack.push(0);
//	active.push_back(0);
//	int index = findGridLocation(x_0, min, nx, ny, dr);
//	grid[index] = 0;
//
//	int count = 0;
//	//while (!activeStack.empty()){
//	while (!active.empty()){
//		count++;
//
//		// pop top index off the active stack
//		//int i = activeStack.top();
//		std::uniform_int_distribution<int> random(0, (int)active.size()-1);
//		int rand = random(generator);
//		int i = active[rand];
//
//		// find the point x_i
//		glm::vec3 x_i = newPoints[i];
//
//		// generate up to k points choosen uniformly from the spherical annulus between radius r and 2r around x_i
//		bool sampleFound = false;
//		for (unsigned int j = 0; j < k; j++){
//			float radius = radius_dist(generator);
//			float theta = theta_dist(generator);
//			float phi = phi_dist(generator);
//
//			float x = x_i.x + radius*sin(theta)*cos(phi);
//			float y = x_i.y + radius*sin(theta)*sin(phi);
//			float z = x_i.z + radius*cos(theta);
//
//			if (x >= min.x && x <= max.x && y >= min.y && y <= max.y && z >= min.z && z <= max.z){
//				glm::vec3 k_point = glm::vec3(x, y, z);
//
//				// for each point in turn, check if it is within distance r of existing points
//				index = findGridLocation(k_point, min, nx, ny, dr);
//
//				if (index >= nx*ny*nz){
//					std::cout << "index: " << index << std::endl;
//				}
//
//				if (grid[index] != -1){
//					continue;
//				}
//
//				if (!IsNearAnotherPoint(grid, newPoints, k_point, index, nx, ny, nz, r)){
//					sampleFound = true;
//					newPoints.push_back(k_point);
//					grid[index] = (int)newPoints.size() - 1;
//					active.push_back((int)newPoints.size() - 1);
//					//activeStack.push((int)newPoints.size() - 1);
//				}
//			}
//		}
//
//		if (sampleFound == false){
//			// swap 
//			active[rand] = active[active.size() - 1];
//			active.resize(active.size() - 1);
//			//activeStack.pop();
//		}
//		
//		if (i % 100 == 0){
//			//std::cout << "count: " << count << " stack size: " << activeStack.size() << " points.size(): " << newPoints.size() << " grid.size(): " << grid.size() << std::endl;
//			std::cout << "count: " << count << " stack size: " << active.size() << " points.size(): " << newPoints.size() << " grid.size(): " << grid.size() << std::endl;
//		}
//	}
//
//	for (unsigned int j = 0; j < newPoints.size(); j++){
//		points.push_back(newPoints[j]);
//	}
//}
//
//
//int ParticleSampler::findGridLocation(glm::vec3 point, glm::vec3 min, int nx, int ny, float dr)
//{
//	return ny * nx * (int)floor((point.z- min.z) / dr) + nx * (int)floor((point.y - min.y) / dr) + (int)floor((point.x - min.x) / dr);
//}
//
//bool ParticleSampler::IsNearAnotherPoint(std::vector<int> &grid, std::vector<glm::vec3> &points, glm::vec3 point, int index, int nx, int ny, int nz, float r)
//{
//	for (int a = -2; a <= 2; a++){
//		for (int b = -2; b <= 2; b++){
//			for (int c = -2; c <= 2; c++){
//				int index2 = index + nx*ny*c + nx*b + a;
//				if (index2 >= 0 && index2 < nx*ny*nz){
//					if (grid[index2] != -1){
//						glm::vec3 otherPoint = points[grid[index2]];
//						float distance = glm::dot(otherPoint - point, otherPoint - point);
//						if (distance < r*r){
//							return true;
//						}
//					}
//				}
//			}
//		}
//	}
//
//	return false;
//}