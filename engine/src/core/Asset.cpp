#include "../../include/core/Asset.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Asset::Asset()
{
	mAssetId = Guid::INVALID;
}

Asset::~Asset()
{

}

Guid Asset::getId() const
{
	return mAssetId;
}