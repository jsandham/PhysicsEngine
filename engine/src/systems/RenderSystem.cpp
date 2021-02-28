#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <unordered_set>

#include "../../include/core/InternalShaders.h"
#include "../../include/core/Intersect.h"
#include "../../include/core/Shader.h"
#include "../../include/core/World.h"

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/core/Input.h"
#include "../../include/core/Log.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem() : System()
{
    mRenderToScreen = true;
}

RenderSystem::RenderSystem(Guid id) : System(id)
{
    mRenderToScreen = true;
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::serialize(std::ostream &out) const
{
    System::serialize(out);
}

void RenderSystem::deserialize(std::istream &in)
{
    System::deserialize(in);
}

void RenderSystem::serialize(YAML::Node& out) const
{
    System::serialize(out);
}

void RenderSystem::deserialize(const YAML::Node& in)
{
    System::deserialize(in);
}

void RenderSystem::init(World *world)
{
    mWorld = world;

    mForwardRenderer.init(mWorld, mRenderToScreen);
    mDeferredRenderer.init(mWorld, mRenderToScreen);
}

void RenderSystem::update(const Input &input, const Time &time)
{
    registerRenderAssets(mWorld);
    registerCameras(mWorld);
    registerLights(mWorld);

    buildRenderObjectsList(mWorld);

    for (size_t i = 0; i < mWorld->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getComponentByIndex<Camera>(i);

        cullRenderObjects(camera);

        buildRenderQueue();
        sortRenderQueue();

        if (camera->mRenderPath == RenderPath::Forward)
        {
            mForwardRenderer.update(input, camera, mRenderQueue, mRenderObjects);
        }
        else
        {
            mDeferredRenderer.update(input, camera, mRenderObjects);
        }
    }
}

void RenderSystem::registerRenderAssets(World *world)
{
    // create all texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Texture2D>(); i++)
    {
        Texture2D *texture = world->getAssetByIndex<Texture2D>(i);
        if (!texture->isCreated())
        {
            texture->create();

            if (!texture->isCreated())
            {
                std::string errorMessage = "Error: Failed to create texture " + texture->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }

        if (texture->updateRequired())
        {
            texture->update();
        }
    }

    // compile all shader assets and configure uniform blocks not already compiled
    std::unordered_set<Guid> shadersCompiledThisFrame;
    for (size_t i = 0; i < world->getNumberOfAssets<Shader>(); i++)
    {
        Shader *shader = world->getAssetByIndex<Shader>(i);

        if (!shader->isCompiled())
        {
            shader->compile();

            if (!shader->isCompiled())
            {
                std::string errorMessage = "Shader failed to compile " + shader->getId().toString() + "\n";
                Log::error(&errorMessage[0]);
            }

            shadersCompiledThisFrame.insert(shader->getId());
        }
    }

    // update material on shader change
    for (size_t i = 0; i < world->getNumberOfAssets<Material>(); i++)
    {
        Material *material = world->getAssetByIndex<Material>(i);

        std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderId());

        if (material->hasShaderChanged() || it != shadersCompiledThisFrame.end())
        {
            material->onShaderChanged(world); // need to also do this if the shader code changed but the assigned shader
                                              // on the material remained the same!
        }
    }

    // create all mesh assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Mesh>(); i++)
    {
        Mesh *mesh = world->getAssetByIndex<Mesh>(i);

        if (!mesh->isCreated())
        {
            mesh->create();

            if (!mesh->isCreated())
            {
                std::string errorMessage = "Error: Failed to create mesh " + mesh->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }
    }
}

void RenderSystem::registerCameras(World *world)
{
    for (size_t i = 0; i < world->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = world->getComponentByIndex<Camera>(i);

        if (!camera->isCreated())
        {
            camera->createTargets();
        }

        /*if (camera->isViewportChanged()) {
            camera->resize();
        }*/
    }
}

void RenderSystem::registerLights(World *world)
{
    for (size_t i = 0; i < world->getNumberOfComponents<Light>(); i++)
    {
        Light *light = world->getComponentByIndex<Light>(i);

        if (!light->isCreated())
        {
            light->createTargets();
        }

        if (light->isShadowMapResolutionChanged())
        {
            light->resizeTargets();
        }
    }
}

void RenderSystem::buildRenderObjectsList(World *world)
{
    mRenderObjects.clear();

    // add enabled renderers to render object list
    for (size_t i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++)
    {
        MeshRenderer *meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

        if (meshRenderer->mEnabled)
        {
            Transform *transform = meshRenderer->getComponent<Transform>(world);
            Mesh *mesh = world->getAssetById<Mesh>(meshRenderer->getMesh());

            if (transform == NULL)
            {
                continue;
            }

            glm::mat4 model = transform->getModelMatrix();

            for (int j = 0; j < meshRenderer->mMaterialCount; j++)
            {
                int materialIndex = world->getIndexOf(meshRenderer->getMaterial(j));
                Material *material = world->getAssetByIndex<Material>(materialIndex);

                int shaderIndex = world->getIndexOf(material->getShaderId());

                int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(j);
                int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(j);

                RenderObject object;
                object.transformId = transform->getId();
                object.meshRendererId = meshRenderer->getId();
                object.meshRendererIndex = (int)i;
                object.materialIndex = materialIndex;
                object.shaderIndex = shaderIndex;
                object.model = model;
                object.start = subMeshVertexStartIndex;
                object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
                object.vao = mesh->getNativeGraphicsVAO();
                object.culled = false;

                mRenderObjects.push_back(object);
            }
        }
    }
}

void RenderSystem::cullRenderObjects(Camera *camera)
{
    int count = 0;
    for (size_t i = 0; i < mRenderObjects.size(); i++)
    {
        glm::vec3 centre = mRenderObjects[i].boundingSphere.mCentre;
        float radius = mRenderObjects[i].boundingSphere.mRadius;

        glm::vec4 temp = mRenderObjects[i].model * glm::vec4(centre.x, centre.y, centre.z, 1.0f);

        Sphere cullingSphere;
        cullingSphere.mCentre = glm::vec3(temp.x, temp.y, temp.z);
        cullingSphere.mRadius = radius;

        if (Intersect::intersect(cullingSphere, camera->getFrustum()))
        {
            count++;
        }
    }
}

void RenderSystem::buildRenderQueue()
{
    mRenderQueue.clear();

    for (size_t i = 0; i < mRenderObjects.size(); i++)
    {
        if (!mRenderObjects[i].culled)
        {
            uint64_t key = 0;

            uint32_t matIndex = mRenderObjects[i].materialIndex;
            uint32_t depth = 2342;
            uint32_t reserved = 112;

            // [ reserved ][  depth  ][ material index]
            // [  8-bits  ][ 24-bits ][    32-bits    ]
            // 64                                     0
            constexpr uint32_t matMask = 0xFFFFFFFF; // 32 least significant bits mask
            constexpr uint32_t depthMask = 0xFFFFFF; // 24 least significant bits mask
            constexpr uint32_t reservedMask = 0xFF;  // 8 least significant bits mask

            matIndex = matMask & matIndex;
            depth = depthMask & depth;
            reserved = reservedMask & reserved;

            uint64_t temp = 0;
            key |= ((reserved | temp) << 56);
            key |= ((depth | temp) << 32); // depth back to front
            key |= ((matIndex | temp) << 0);

            mRenderQueue.push_back(std::make_pair(key, (int)i));
        }
    }
}

void RenderSystem::sortRenderQueue()
{
    // sort render queue from highest priority key to lowest
    std::sort(mRenderQueue.begin(), mRenderQueue.end(),
              [=](std::pair<uint64_t, int> &a, std::pair<uint64_t, int> &b) { return a.first > b.first; });
}