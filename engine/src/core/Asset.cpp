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

bool Asset::isInternal(int type)
{
	return type >= PhysicsEngine::MIN_INTERNAL_ASSET && type <= PhysicsEngine::MAX_INTERNAL_ASSET;
}