#include "../../include/systems/GridRendererSystem.h"

using namespace PhysicsEngine;

GridRendererSystem::GridRendererSystem()
{

}

GridRendererSystem::~GridRendererSystem()
{

}

std::vector<char> GridRendererSystem::serialize() const
{
	return serialize(systemId);
}

std::vector<char> GridRendererSystem::serialize(Guid systemId) const
{
	std::vector<char> temp;
	return temp;
}

void GridRendererSystem::deserialize(std::vector<char> data)
{

}

void GridRendererSystem::init(World* world)
{

}

void GridRendererSystem::update(Input input)
{

}

