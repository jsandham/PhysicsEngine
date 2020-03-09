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

Guid Asset::getId() const
{
	return assetId;
}