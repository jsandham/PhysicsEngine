#ifndef __PARTICLE_SAMPLER_H__
#define __PARTICLE_SAMPLER_H__

#include <vector>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class ParticleSampler
	{
		public:
			static void poissonSampler(std::vector<float> &points, float minx, float miny, float minz, float maxx, float maxy, float maxz, float h, float r, unsigned int k);
			static void randomParticles(std::vector<float> &points, float minx, float miny, float minz, float maxx, float maxy, float maxz, float h, int numPoints);
			static void gridOfParticlesXYPlane(std::vector<float> &points, float dx, float dy, float z, int nx, int ny);
			static void gridOfParticlesXZPlane(std::vector<float> &points, float dx, float dz, float y, int nx, int nz);

		private:
			static int findGridLocation(float x, float y, float z, float minx, float miny, float minz, int nx, int ny, float dr);
			static bool IsNearAnotherPoint(std::vector<int> &grid, std::vector<float> &points, float x, float y, float z, int index, int nx, int ny, int nz, float r);
	};
}

#endif