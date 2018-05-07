#include "Cloth.h"
#include "../core/Input.h"

using namespace PhysicsEngine;

// Cloth::Cloth()
// {
// 	physics = new CudaClothPhysics();
// }

// Cloth::Cloth(Entity *entity)
// {
// 	this->entity = entity;
// }

// Cloth::~Cloth()
// {
// 	delete physics;
// }

// void Cloth::init()
// {
// 	physics->init();

// 	run = false;
// }

// void Cloth::update()
// {
// 	if (Input::getKeyDown(sf::Keyboard::Return)){
// 		run = true;
// 	}

// 	if (run){
// 		physics->update();
// 	}
// }

// void Cloth::setParticles(std::vector<float> &particles)
// {
// 	physics->setParticles(particles);
// }

// void Cloth::setParticleTypes(std::vector<int> &particleTypes)
// {
// 	physics->setParticleTypes(particleTypes);
// }

// void Cloth::setNx(int nx)
// {
// 	physics->setNx(nx);
// }

// void Cloth::setNy(int ny)
// {
// 	physics->setNy(ny);
// }

// std::vector<float>& Cloth::getParticles()
// {
// 	return physics->getParticles();
// }

// std::vector<int>& Cloth::getParticleTypes()
// {
// 	return physics->getParticleTypes();
// }

// int Cloth::getNx()
// {
// 	return physics->getNx();
// }

// int Cloth::getNy()
// {
// 	return physics->getNy();
// }