#include "../../include/core/Asset.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Asset::Asset()
{
	assetId = Guid::INVALID;
}

Asset::~Asset()
{

}

void Asset::setManager(World* world)
{
	this->world = world;
}