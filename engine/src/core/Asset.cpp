#include "../../include/core/Asset.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Asset::Asset()
{
	assetId = Guid::INVALID;
}

Asset::~Asset()
{

}

void Asset::setManager(Manager* manager)
{
	this->manager = manager;
}