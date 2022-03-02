#include "../../include/systems/TerrainSystem.h"
#include "../../include/components/Transform.h"
#include "../../include/components/Terrain.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

TerrainSystem::TerrainSystem(World *world) : System(world)
{

}

TerrainSystem::TerrainSystem(World *world, const Guid& id) : System(world, id)
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
            float x = transform->mPosition.x;
            float z = transform->mPosition.z;

            Rect centreChunkRect = terrain->getCentreChunkRect();
            
            if (!centreChunkRect.contains(x, z))
            {
                float dx = 0.0f;
                float dz = 0.0f;
                if (x < centreChunkRect.mX)
                {
                    dx += -10.0f;
                }
                else if (x > centreChunkRect.mX + centreChunkRect.mWidth)
                {
                    dx += 10.0f;
                }

                if (z < centreChunkRect.mY)
                {
                    dz += -10.0f;
                }
                else if (z > centreChunkRect.mY + centreChunkRect.mHeight)
                {
                    dz += 10.0f;
                }

                terrain->updateTerrainHeight(dx, dz);
            }
        }
    }
}