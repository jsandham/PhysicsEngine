#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <unordered_set>

#define GLM_FORCE_RADIANS

#include "glm/gtc/matrix_transform.hpp"

#include "../../include/core/Intersect.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem(World *world, const Id &id) : System(world, id)
{
}

RenderSystem::RenderSystem(World *world, const Guid &guid, const Id &id) : System(world, guid, id)
{
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void RenderSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int RenderSystem::getType() const
{
    return PhysicsEngine::RENDERSYSTEM_TYPE;
}

std::string RenderSystem::getObjectName() const
{
    return PhysicsEngine::RENDERSYSTEM_NAME;
}

void RenderSystem::init(World *world)
{
    mWorld = world;

    mForwardRenderer.init(mWorld);
    mDeferredRenderer.init(mWorld);
    mDebugRenderer.init(mWorld);
}

void RenderSystem::update(const Input &input, const Time &time)
{
    registerRenderAssets(mWorld);
    registerCameras(mWorld);
    registerLights(mWorld);

    buildRenderObjectsList(mWorld);
    buildSpriteObjectsList(mWorld);

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(i);

        if (camera->mEnabled)
        {
            cullRenderObjects(camera);

            buildRenderQueue();
            sortRenderQueue();

            if (camera->mColorTarget == ColorTarget::Color || camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                if (camera->mRenderPath == RenderPath::Forward)
                {
                    mForwardRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds, mSpriteObjects);
                }
                else
                {
                    mDeferredRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds);
                }
            }
            else
            {
                mDebugRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds);
            }
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
                std::string errorMessage = "Error: Failed to create texture " + texture->getGuid().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }

        if (texture->updateRequired())
        {
            texture->update();
        }
    }

    // create all render texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<RenderTexture>(); i++)
    {
        RenderTexture* texture = world->getAssetByIndex<RenderTexture>(i);
        if (!texture->isCreated())
        {
            texture->create();

            if (!texture->isCreated())
            {
                std::string errorMessage = "Error: Failed to create render texture " + texture->getGuid().toString() + "\n";
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
            shader->preprocess();
            shader->compile();

            if (!shader->isCompiled())
            {
                std::string errorMessage = "Shader failed to compile " + shader->getName() + " " + shader->getGuid().toString() + "\n";
                Log::error(&errorMessage[0]);
            }

            shadersCompiledThisFrame.insert(shader->getGuid());
        }
    }

    // update material on shader change
    for (size_t i = 0; i < world->getNumberOfAssets<Material>(); i++)
    {
        Material *material = world->getAssetByIndex<Material>(i);

        std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderId());

        if (material->hasShaderChanged() || it != shadersCompiledThisFrame.end())
        {
            material->onShaderChanged(); // need to also do this if the shader code changed but the assigned shader
                                              // on the material remained the same!
        }

        if (material->hasTextureChanged())
        {
            material->onTextureChanged();
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
                std::string errorMessage = "Error: Failed to create mesh " + mesh->getGuid().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }
    }

    // create all sprite assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Sprite>(); i++)
    {
        Sprite* sprite = world->getAssetByIndex<Sprite>(i);

        if (!sprite->isCreated())
        {
            sprite->create();

            if (!sprite->isCreated())
            {
                std::string errorMessage = "Error: Failed to create sprite " + sprite->getGuid().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }
    }
}

void RenderSystem::registerCameras(World *world)
{
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = world->getActiveScene()->getComponentByIndex<Camera>(i);

        if (!camera->isCreated())
        {
            camera->createTargets();
        }
    }
}

void RenderSystem::registerLights(World *world)
{
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Light>(); i++)
    {
        Light *light = world->getActiveScene()->getComponentByIndex<Light>(i);

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
    mTotalRenderObjects.clear();
    mTotalModels.clear();
    mTotalTransformIds.clear();
    mTotalBoundingSpheres.clear();

    InstanceMap instanceMap;

    // add enabled renderers to render object list
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<MeshRenderer>(); i++)
    {
        MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);

        if (meshRenderer != nullptr && meshRenderer->mEnabled)
        {
            Transform *transform = meshRenderer->getComponent<Transform>();
            Mesh *mesh = world->getAssetByGuid<Mesh>(meshRenderer->getMesh());

            if (transform == nullptr || mesh == nullptr){ continue; }

            glm::mat4 model = transform->getModelMatrix();

            Sphere boundingSphere = computeWorldSpaceBoundingSphere(model, mesh->getBounds());

            for (int j = 0; j < meshRenderer->mMaterialCount; j++)
            {
                int materialIndex = world->getIndexOf(meshRenderer->getMaterial(j));
                Material *material = world->getAssetByIndex<Material>(materialIndex);

                // could be nullptr if for example we are adding a material to the renderer in the editor
                // but we have not yet actually set the material
                if (material == nullptr){ break; }

                int shaderIndex = world->getIndexOf(material->getShaderId());

                int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(j);
                int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(j);

                if (material->mEnableInstancing)
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = subMeshVertexStartIndex;
                    object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
                    object.vao = mesh->getNativeGraphicsVAO();
                    object.instanceModelVbo = *reinterpret_cast<unsigned int*>(mesh->getNativeGraphicsVBO(MeshVBO::InstanceModel));
                    object.instanceColorVbo = *reinterpret_cast<unsigned int*>(mesh->getNativeGraphicsVBO(MeshVBO::InstanceColor));
                    object.instanced = true;

                    std::pair<Guid, RenderObject> key = std::make_pair(material->getGuid(), object);

                    auto it = instanceMap.find(key);
                    if(it != instanceMap.end())
                    {
                        it->second.models.push_back(model);
                        it->second.transformIds.push_back(transform->getId());
                        it->second.boundingSpheres.push_back(boundingSphere);
                    }
                    else
                    {
                        instanceMap[key].models = std::vector<glm::mat4>();
                        instanceMap[key].transformIds = std::vector<Id>();
                        instanceMap[key].boundingSpheres = std::vector<Sphere>();

                        instanceMap[key].models.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
                        instanceMap[key].transformIds.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
                        instanceMap[key].boundingSpheres.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
                       
                        instanceMap[key].models.push_back(model);
                        instanceMap[key].transformIds.push_back(transform->getId());
                        instanceMap[key].boundingSpheres.push_back(boundingSphere);
                    }
                }
                else
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = subMeshVertexStartIndex;
                    object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
                    object.vao = mesh->getNativeGraphicsVAO();
                    object.instanceModelVbo = -1;
                    object.instanceColorVbo = -1;
                    object.instanced = false;

                    mTotalRenderObjects.push_back(object);
                    mTotalModels.push_back(model);
                    mTotalTransformIds.push_back(transform->getId());
                    mTotalBoundingSpheres.push_back(boundingSphere);
                }
            }
        }
    }

    for (auto it = instanceMap.begin(); it != instanceMap.end(); it++)
    {
        std::vector<glm::mat4> &models = it->second.models;
        std::vector<Id> &transformIds = it->second.transformIds;
        std::vector<Sphere> &boundingSpheres = it->second.boundingSpheres;

        size_t count = 0;
        while (count < models.size())
        {
            RenderObject renderObject = it->first.second;
            renderObject.instanceStart = count;
            renderObject.instanceCount = std::min(models.size() - count, static_cast<size_t>(Renderer::getRenderer()->INSTANCE_BATCH_SIZE));
            
            mTotalRenderObjects.push_back(renderObject);

            size_t start = mTotalModels.size();
            mTotalModels.resize(mTotalModels.size() + renderObject.instanceCount);
            mTotalTransformIds.resize(mTotalTransformIds.size() + renderObject.instanceCount);
            mTotalBoundingSpheres.resize(mTotalBoundingSpheres.size() + renderObject.instanceCount);

            for (size_t i = 0; i < renderObject.instanceCount; i++)
            {
                /*mTotalModels.push_back(models[renderObject.instanceStart + i]);
                mTotalTransformIds.push_back(transformIds[renderObject.instanceStart + i]);
                mTotalBoundingSpheres.push_back(boundingSpheres[renderObject.instanceStart + i]);*/
                mTotalModels[start + i] = models[renderObject.instanceStart + i];
                mTotalTransformIds[start + i] = transformIds[renderObject.instanceStart + i];
                mTotalBoundingSpheres[start + i] = boundingSpheres[renderObject.instanceStart + i];
            }
       
            count += Renderer::getRenderer()->INSTANCE_BATCH_SIZE;
        }
    }

    assert(mTotalModels.size() == mTotalTransformIds.size());
    assert(mTotalModels.size() == mTotalBoundingSpheres.size());

    // add enabled terrain to render object list
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = world->getActiveScene()->getComponentByIndex<Terrain>(i);

        if (terrain != nullptr)
        {
            Transform *transform = terrain->getComponent<Transform>();

            if (transform == nullptr){ continue; }

            glm::mat4 model = transform->getModelMatrix();

            int materialIndex = world->getIndexOf(terrain->getMaterial());
            Material *material = world->getAssetByIndex<Material>(materialIndex);

            // could be nullptr if for example we are adding a material to the renderer in the editor
            // but we have not yet actually set the material
            if (material == nullptr){ break; }

            int shaderIndex = world->getIndexOf(material->getShaderId());
 
            for (int j = 0; j < terrain->getTotalChunkCount(); j++)
            {
                if (terrain->isChunkEnabled(j))
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = terrain->getChunkStart(j);
                    object.size = terrain->getChunkSize(j);
                    object.vao = terrain->getNativeGraphicsVAO();
                    object.instanceModelVbo = -1;
                    object.instanceColorVbo = -1;
                    object.instanced = false;

                    mTotalRenderObjects.push_back(object);
                    mTotalModels.push_back(model);
                    mTotalTransformIds.push_back(transform->getId());

                    mTotalBoundingSpheres.push_back(computeWorldSpaceBoundingSphere(model, terrain->getChunkBounds(j)));
                }
            }
        }
    }

    assert(mTotalModels.size() == mTotalTransformIds.size());
    assert(mTotalModels.size() == mTotalBoundingSpheres.size());

    mWorld->mBoundingSpheres = mTotalBoundingSpheres;
}

void RenderSystem::buildSpriteObjectsList(World* world)
{
    mSpriteObjects.clear();

    // add enabled renderers to render object list
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<SpriteRenderer>(); i++)
    {
        SpriteRenderer *spriteRenderer = world->getActiveScene()->getComponentByIndex<SpriteRenderer>(i);

        if (spriteRenderer->mEnabled)
        {
            Transform* transform = spriteRenderer->getComponent<Transform>();
            Sprite *sprite = world->getAssetByGuid<Sprite>(spriteRenderer->getSprite());

            if (transform == nullptr || sprite == nullptr)
            {
                continue;
            }

            Texture2D *texture = world->getAssetByGuid<Texture2D>(sprite->getTextureId());

            glm::vec2 size = glm::vec2(100, 100);
            //float rotate = 0.0f;

            glm::mat4 model = transform->getModelMatrix();

            if (spriteRenderer->mFlipX)
            {
                model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0, 1, 0));
            }

            if (spriteRenderer->mFlipY)
            {
                model = glm::rotate(model, glm::radians(180.0f), glm::vec3(1, 0, 0));
            }

            SpriteObject object;
            object.model = model;
            object.color = spriteRenderer->mColor;
            object.vao = sprite->getNativeGraphicsVAO();

            if (texture != nullptr)
            {
                object.texture = *reinterpret_cast<unsigned int*>(texture->getNativeGraphics());
            }
            else
            {
                object.texture = -1;
            }

            mSpriteObjects.push_back(object);
        }
    }
}

void RenderSystem::cullRenderObjects(Camera *camera)
{
    mModels.resize(mTotalModels.size());
    mTransformIds.resize(mTotalTransformIds.size());
    mRenderObjects.resize(mTotalRenderObjects.size());

    int index = 0;
    int objectCount = 0;
    int count = 0;
    for (size_t i = 0; i < mTotalRenderObjects.size(); i++)
    {
        //dont perform any culling on instanced objects
        if (mTotalRenderObjects[i].instanced)
        {
            mRenderObjects[objectCount] = mTotalRenderObjects[i];
            mRenderObjects[objectCount].instanceStart = count; 

            for (size_t j = 0; j < mTotalRenderObjects[i].instanceCount; j++)
            {
                mModels[count + j] = mTotalModels[index];
                mTransformIds[count + j] = mTotalTransformIds[index];
              
                index++;
            }

            objectCount++;
            count += mTotalRenderObjects[i].instanceCount;
        }
        else
        {
            if (Intersect::intersect(mTotalBoundingSpheres[index], camera->getFrustum()))
            {
                mRenderObjects[objectCount] = mTotalRenderObjects[i];

                mModels[count] = mTotalModels[index];
                mTransformIds[count] = mTotalTransformIds[index];
           
                objectCount++;
                count++;
            }

            index++;
        }
    }

    mModels.resize(count);
    mTransformIds.resize(count);
    mRenderObjects.resize(objectCount);
}

void RenderSystem::buildRenderQueue()
{
    //mRenderQueue.clear();

    //for (size_t i = 0; i < mTotalRenderObjects.size(); i++)
    //{
    //    // for now dont sort
    //    uint64_t key = i;

    //    mRenderQueue.push_back(std::make_pair(key, (int)i));

    //    //if (!mTotalRenderObjects[i].culled)
    //    //{
    //    //    uint64_t key = 0;

    //    //    uint32_t matIndex = mTotalRenderObjects[i].materialIndex;
    //    //    uint32_t depth = 2342;
    //    //    uint32_t reserved = 112;

    //    //    // [ reserved ][  depth  ][ material index]
    //    //    // [  8-bits  ][ 24-bits ][    32-bits    ]
    //    //    // 64                                     0
    //    //    constexpr uint32_t matMask = 0xFFFFFFFF; // 32 least significant bits mask
    //    //    constexpr uint32_t depthMask = 0xFFFFFF; // 24 least significant bits mask
    //    //    constexpr uint32_t reservedMask = 0xFF;  // 8 least significant bits mask

    //    //    matIndex = matMask & matIndex;
    //    //    depth = depthMask & depth;
    //    //    reserved = reservedMask & reserved;

    //    //    uint64_t temp = 0;
    //    //    key |= ((reserved | temp) << 56);
    //    //    key |= ((depth | temp) << 32); // depth back to front
    //    //    key |= ((matIndex | temp) << 0);

    //    //    mRenderQueue.push_back(std::make_pair(key, (int)i));
    //    //}
    //}
}

void RenderSystem::sortRenderQueue()
{
    // sort render queue from highest priority key to lowest
    //std::sort(mRenderQueue.begin(), mRenderQueue.end(),
    //          [=](std::pair<uint64_t, int> &a, std::pair<uint64_t, int> &b) { return a.first > b.first; });
}



Sphere RenderSystem::computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere)
{
    glm::vec4 temp = model * glm::vec4(sphere.mRadius, sphere.mRadius, sphere.mRadius, 0.0f);

    Sphere boundingSphere;
    boundingSphere.mCentre = glm::vec3(model * glm::vec4(sphere.mCentre, 1.0f));
    boundingSphere.mRadius = std::max(temp.x, std::max(temp.y, temp.z));

    return boundingSphere;
}