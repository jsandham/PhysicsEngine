#include "../../include/systems/TerrainSystem.h"
#include "../../include/components/Transform.h"
#include "../../include/components/Terrain.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

TerrainSystem::TerrainSystem(World *world) : System(world)
{

}

TerrainSystem::TerrainSystem(World *world, Guid id) : System(world, id)
{

}

TerrainSystem::~TerrainSystem()
{

}

void TerrainSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void TerrainSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int TerrainSystem::getType() const
{
    return PhysicsEngine::TERRAINSYSTEM_TYPE;
}

std::string TerrainSystem::getObjectName() const
{
    return PhysicsEngine::TERRAINSYSTEM_NAME;
}

void TerrainSystem::init(World *world)
{
    mWorld = world;
}

void TerrainSystem::update(const Input &input, const Time &time)
{
    for (size_t i = 0; i < mWorld->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = mWorld->getComponentByIndex<Terrain>(i);

        if (!terrain->isCreated())
        {
            terrain->generateTerrain();
        }

        Transform *transform = mWorld->getComponentById<Transform>(terrain->mCameraTransformId);

        if (transform != nullptr)
        {
            
        }
    }
}