#include "../../include/systems/TerrainSystem.h"
#include "../../include/components/Terrain.h"
#include "../../include/components/Transform.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

TerrainSystem::TerrainSystem(World *world, const Id &id) : System(world, id)
{
}

TerrainSystem::TerrainSystem(World *world, const Guid &guid, const Id &id) : System(world, guid, id)
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
    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = mWorld->getActiveScene()->getComponentByIndex<Terrain>(i);

        if (!terrain->isCreated())
        {
            terrain->generateTerrain();
        }

        Transform *transform = mWorld->getActiveScene()->getComponentByGuid<Transform>(terrain->mCameraTransformId);

        if (transform != nullptr)
        {
            float x = transform->getPosition().x;
            float z = transform->getPosition().z;

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