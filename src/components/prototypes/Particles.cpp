#include <iostream>
#include "Particles.h"

using namespace PhysicsEngine;

// Particles::Particles()
// {
// 	physics = new CudaDEMPhysics();
// }

// Particles::Particles(Entity *entity)
// {
// 	this->entity = entity;

// 	physics = new CudaDEMPhysics();
// }

// Particles::~Particles()
// {
// 	delete physics;
// }

// void Particles::init()
// {
// 	std::cout << "Particles init called" << std::endl;
// 	physics->init();
// }

// void Particles::update()
// {
// 	physics->update();
// }

// void Particles::setGrid(VoxelGrid *grid)
// {
// 	voxelGrid = grid;

// 	physics->setGridDomain(grid);
// }

// void Particles::setParticles(std::vector<float> &particles)
// {
// 	physics->setParticles(particles);
// }

// void Particles::setParticleTypes(std::vector<int> &particleTypes)
// {
// 	physics->setParticleTypes(particleTypes);
// }

// VoxelGrid* Particles::getGrid()
// {
// 	return voxelGrid;
// }

// std::vector<float>& Particles::getParticles()
// {
// 	return physics->getParticles();
// }

// std::vector<int>& Particles::getParticleTypes()
// {
// 	return physics->getParticleTypes();
// }