#include "../../include/systems/TerrainSystem.h"

#include "../../include/components/Terrain.h"
#include "../../include/components/Transform.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

TerrainSystem::TerrainSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

TerrainSystem::TerrainSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

TerrainSystem::~TerrainSystem()
{
}

void TerrainSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void TerrainSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int TerrainSystem::getType() const
{
    return PhysicsEngine::TERRAINSYSTEM_TYPE;
}

std::string TerrainSystem::getObjectName() const
{
    return PhysicsEngine::TERRAINSYSTEM_NAME;
}

Guid TerrainSystem::getGuid() const
{
    return mGuid;
}

Id TerrainSystem::getId() const
{
    return mId;
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