#include <iostream>
#include "Fluid.h"
#include "../core/Input.h"

using namespace PhysicsEngine;

Fluid::Fluid()
{
	//std::cout << "fluid particles constructor called" << std::endl;
	//physics = new CudaFluidPhysics();
}

Fluid::Fluid(Entity *entity)
{
	this->entity = entity;

	//std::cout << "fluid particles constructor called" << std::endl;
	//physics = new CudaFluidPhysics();
}

Fluid::~Fluid()
{
	//std::cout << "FluidParticles destructor called" << std::endl;
	//delete physics;
}

// void Fluid::init()
// {
// 	std::cout << "FluidParticles init called" << std::endl;
// 	physics->init();

// 	run = false;
// }

// void Fluid::update()
// {
// 	if (Input::getKeyDown(sf::Keyboard::Return)){
// 		run = true;
// 	}

// 	if (run){
// 		physics->update();
// 	}
// }

// void Fluid::setGrid(VoxelGrid *grid)
// {
// 	voxelGrid = grid;

// 	physics->setGridDomain(grid);
// }

// void Fluid::setParticles(std::vector<float> &particles)
// {
// 	physics->setParticles(particles);
// }

// void Fluid::setParticleTypes(std::vector<int> &particleTypes)
// {
// 	physics->setParticleTypes(particleTypes);
// }

// VoxelGrid* Fluid::getGrid()
// {
// 	return voxelGrid;
// }

// std::vector<float>& Fluid::getParticles()
// {
// 	return physics->getParticles();
// }

// std::vector<int>& Fluid::getParticleTypes()
// {
// 	return physics->getParticleTypes();
// }