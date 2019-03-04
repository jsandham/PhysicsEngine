#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>

#include "Component.h"

// #include "ParticlePhysics.h"
// #include "../CudaFluidPhysics.cuh"

// #include "../VoxelGrid.h"

namespace PhysicsEngine
{
	class Fluid : public Component
	{
		private:
			bool run;

			// VoxelGrid *voxelGrid;
			// CudaFluidPhysics *physics;

		public:
			Fluid();
			Fluid(std::vector<char> data);
			~Fluid();

			// void init();
			// void update();

			// void setGrid(VoxelGrid *grid);
			// void setParticles(std::vector<float> &particles);
			// void setParticleTypes(std::vector<int> &particleTypes);

			// VoxelGrid* getGrid();
			// std::vector<float>& getParticles();
			// std::vector<int>& getParticleTypes();
	};
}

#endif