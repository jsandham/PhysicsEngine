#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <unordered_set>

#define GLM_FORCE_RADIANS

#include "glm/gtc/matrix_transform.hpp"

#include "../../include/systems/RenderSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Intersect.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

RenderSystem::RenderSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void RenderSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int RenderSystem::getType() const
{
    return PhysicsEngine::RENDERSYSTEM_TYPE;
}

std::string RenderSystem::getObjectName() const
{
    return PhysicsEngine::RENDERSYSTEM_NAME;
}

Guid RenderSystem::getGuid() const
{
    return mGuid;
}

Id RenderSystem::getId() const
{
    return mId;
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

    cacheRenderData(mWorld);

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(i);

        if (camera->mEnabled)
        {
            frustumCulling(mWorld, camera);

            buildRenderObjectsList(mWorld, camera);

            buildRenderQueue();
            sortRenderQueue();


            // batching after sort?


            if (camera->mColorTarget == ColorTarget::Color || camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                if (camera->mRenderPath == RenderPath::Forward)
                {
                    mForwardRenderer.update(input, camera, mDrawCalls, mModels, mTransformIds);
                }
                else
                {
                    mDeferredRenderer.update(input, camera, mDrawCalls, mModels, mTransformIds);
                }
            }
            else
            {
                mDebugRenderer.update(input, camera, mDrawCalls, mModels, mTransformIds);
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
        if (texture->deviceUpdateRequired())
        {
            texture->copyTextureToDevice();
        }

        if (texture->updateRequired())
        {
            texture->updateTextureParameters();
        }
    }

    // create all render texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<RenderTexture>(); i++)
    {
        RenderTexture *texture = world->getAssetByIndex<RenderTexture>(i);
        if (texture->deviceUpdateRequired())
        {
            texture->copyTextureToDevice();
        }

        if (texture->updateRequired())
        {
            texture->updateTextureParameters();
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
                std::string errorMessage =
                    "Shader failed to compile " + shader->mName + " " + shader->getGuid().toString() + "\n";
                Log::error(&errorMessage[0]);
            }

            shadersCompiledThisFrame.insert(shader->getGuid());
        }
    }

    // update material on shader change
    for (size_t i = 0; i < world->getNumberOfAssets<Material>(); i++)
    {
        Material *material = world->getAssetByIndex<Material>(i);

        std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderGuid());

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

        if (mesh->deviceUpdateRequired())
        {
            mesh->copyMeshToDevice();
        }
    }
}

void RenderSystem::cacheRenderData(World *world)
{
    size_t meshRendererCount = world->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    mCachedModels.resize(meshRendererCount);
    mCachedTransformIds.resize(meshRendererCount);
    mCachedBoundingSpheres.resize(meshRendererCount);
    mCachedMeshIndices.resize(meshRendererCount);
    mCachedMaterialIndices.resize(8 * meshRendererCount);

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        TransformData *transformData = world->getActiveScene()->getTransformDataByMeshRendererIndex(i);
        assert(transformData != nullptr);

        mCachedModels[i] = transformData->getModelMatrix();
    }

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        size_t transformIndex = world->getActiveScene()->getIndexOfTransformFromMeshRendererIndex(i);
        Transform *transform = world->getActiveScene()->getComponentByIndex<Transform>(transformIndex);
        assert(transform != nullptr);

        mCachedTransformIds[i] = transform->getId();
    }

    Id lastMeshId = Id::INVALID;
    int lastMeshIndex = -1;

    Id lastMaterialId = Id::INVALID;
    int lastMaterialIndex = -1;

    glm::vec3 boundingVolumeCentre = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 boundingVolumeMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 boundingVolumeMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
    for (size_t i = 0; i < meshRendererCount; i++)
    {
        MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
        assert(meshRenderer != nullptr);

        Id meshId = meshRenderer->getMeshId();
        if (meshId != lastMeshId)
        {
            lastMeshIndex = world->getIndexOf(meshId);
            lastMeshId = meshId;
        }

        mCachedMeshIndices[i] = lastMeshIndex;

        for (int j = 0; j < meshRenderer->mMaterialCount; j++)
        {
            Id materialId = meshRenderer->getMaterialId(j);
            if (materialId != lastMaterialId)
            {
                lastMaterialIndex = world->getIndexOf(materialId);
                lastMaterialId = materialId;
            }

            mCachedMaterialIndices[8 * i + j] = lastMaterialIndex;
        }

        Mesh *mesh = world->getAssetByIndex<Mesh>(lastMeshIndex);

        mCachedBoundingSpheres[i] = computeWorldSpaceBoundingSphere(mCachedModels[i], mesh->getBounds());
        
        glm::vec3 centre = mCachedBoundingSpheres[i].mCentre;
        float radius = mCachedBoundingSpheres[i].mRadius;

        boundingVolumeMin.x = glm::min(boundingVolumeMin.x, centre.x - radius);
        boundingVolumeMin.y = glm::min(boundingVolumeMin.y, centre.y - radius);
        boundingVolumeMin.z = glm::min(boundingVolumeMin.z, centre.z - radius);

        boundingVolumeMax.x = glm::max(boundingVolumeMax.x, centre.x + radius);
        boundingVolumeMax.y = glm::max(boundingVolumeMax.y, centre.y + radius);
        boundingVolumeMax.z = glm::max(boundingVolumeMax.z, centre.z + radius);
    }

    mCachedBoundingVolume.mCentre = 0.5f * (boundingVolumeMax + boundingVolumeMin);
    mCachedBoundingVolume.mSize = (boundingVolumeMax - boundingVolumeMin);
}

void RenderSystem::frustumCulling(World *world, Camera *camera)
{
    size_t meshRendererCount = world->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    mFrustumVisible.resize(meshRendererCount);

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        mFrustumVisible[i] = Intersect::intersect(mCachedBoundingSpheres[i], camera->getFrustum());
    }
}

void RenderSystem::buildRenderObjectsList(World *world, Camera* camera)
{
    size_t meshRendererCount = world->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    std::unordered_map<uint64_t, std::vector<size_t>> instanceMapping;

    // allow up to 8 materials (and hence draw calls) per mesh renderer
    mDrawCallScratch.resize(8 * meshRendererCount);
    mDrawCallMeshRendererIndices.resize(8 * meshRendererCount);
    
    size_t drawCallCount = 0;
    size_t instancedDrawCallCount = 0;
    size_t dataArraysCount = 0;

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        if (mFrustumVisible[i])
        {
            MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
            assert(meshRenderer != nullptr);

            if (meshRenderer->mEnabled)
            {
                // could be nullptr if for example we are adding a mesh to the renderer in the editor
                // but we have not yet actually set the mesh
                if (mCachedMeshIndices[i] != -1)
                {
                    for (int j = 0; j < meshRenderer->mMaterialCount; j++)
                    {
                        int materialIndex = mCachedMaterialIndices[8 * i + j];
                        Material *material = world->getAssetByIndex<Material>(materialIndex);

                        // could be nullptr if for example we are adding a material to the renderer in the editor
                        // but we have not yet actually set the material
                        if (material != nullptr)
                        {
                            int shaderIndex = world->getIndexOf(material->getShaderGuid());

                            if (material->mEnableInstancing)
                            {
                                uint64_t key = generateDrawCall(materialIndex, mCachedMeshIndices[i], shaderIndex, j, 2);

                                auto it = instanceMapping.find(key);
                                if (it != instanceMapping.end())
                                {
                                    if (it->second.size() % Renderer::getRenderer()->INSTANCE_BATCH_SIZE == 0)
                                    {
                                        instancedDrawCallCount++;
                                    }

                                    it->second.push_back(i);
                                }
                                else
                                {
                                    instanceMapping[key] = std::vector<size_t>();
                                    instanceMapping[key].reserve(1000);
                                    instanceMapping[key].push_back(i);
                                    instancedDrawCallCount++;
                                }
                                dataArraysCount++;
                            }
                            else
                            {
                                mDrawCallScratch[drawCallCount].key =
                                    generateDrawCall(materialIndex, mCachedMeshIndices[i], shaderIndex, j, 1);
                                mDrawCallMeshRendererIndices[drawCallCount] = i;
                                drawCallCount++;
                                dataArraysCount++;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    assert(dataArraysCount >= (drawCallCount + instancedDrawCallCount));

    mDrawCalls.resize(drawCallCount + instancedDrawCallCount);
    mModels.resize(dataArraysCount);
    mTransformIds.resize(dataArraysCount);
    mBoundingSpheres.resize(dataArraysCount);

    // single draw calls
    for (size_t i = 0; i < drawCallCount; i++)
    {
        mDrawCalls[i] = mDrawCallScratch[i];
        mModels[i] = mCachedModels[mDrawCallMeshRendererIndices[i]];
        mTransformIds[i] = mCachedTransformIds[mDrawCallMeshRendererIndices[i]];
        mBoundingSpheres[i] = mCachedBoundingSpheres[mDrawCallMeshRendererIndices[i]];
    }

    // instanced draw calls
    size_t index = drawCallCount;
    size_t offset = index;
    for (auto it = instanceMapping.begin(); it != instanceMapping.end(); it++)
    {
        size_t count = 0;
        while (count < it->second.size())
        {
            assert(index < mDrawCalls.size());

            mDrawCalls[index].key = it->first;
            mDrawCalls[index].instanceCount =
                std::min(it->second.size() - count, static_cast<size_t>(Renderer::getRenderer()->INSTANCE_BATCH_SIZE));

            for (size_t i = 0; i < mDrawCalls[index].instanceCount; i++)
            {
                assert(offset + i < mModels.size());
                assert(offset + i < mTransformIds.size());
                assert(offset + i < mBoundingSpheres.size());

                assert(count + i < it->second.size());
                assert(count + i < it->second.size());
                assert(count + i < it->second.size());

                mModels[offset + i] = mCachedModels[it->second[count + i]];
                mTransformIds[offset + i] = mCachedTransformIds[it->second[count + i]];
                mBoundingSpheres[offset + i] = mCachedBoundingSpheres[it->second[count + i]];
            }

            offset += mDrawCalls[index].instanceCount;
            index++;
            count += Renderer::getRenderer()->INSTANCE_BATCH_SIZE;
        }
    }

    assert(index == mDrawCalls.size());
    assert(offset == mModels.size());
    assert(offset == mTransformIds.size());
    assert(offset == mBoundingSpheres.size());
   
































    


























    // add enabled terrain to render object list
    /*for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = world->getActiveScene()->getComponentByIndex<Terrain>(i);

        Transform *transform = terrain->getComponent<Transform>();

        glm::mat4 model = transform->getModelMatrix();

        int materialIndex = world->getIndexOf(terrain->getMaterial());
        Material *material = world->getAssetByIndex<Material>(materialIndex);

        // could be nullptr if for example we are adding a material to the renderer in the editor
        // but we have not yet actually set the material
        if (material == nullptr)
        {
            int shaderIndex = world->getIndexOf(material->getShaderGuid());

            for (int j = 0; j < terrain->getTotalChunkCount(); j++)
            {
                if (terrain->isChunkEnabled(j))
                {
                    RenderObject object;
                    object.key = generateDrawCall(materialIndex, i, shaderIndex, j, 4);
                    object.instanceCount = 0;
                    // object.materialIndex = materialIndex;
                    // object.shaderIndex = shaderIndex;
                    // object.start = terrain->getChunkStart(j);
                    // object.size = terrain->getChunkSize(j);
                    // object.meshHandle = terrain->getNativeGraphicsHandle();
                    // object.instanceModelBuffer = nullptr;
                    // object.instanceColorBuffer = nullptr;
                    // object.instanced = false;
                    // object.indexed = false;

                    mTotalRenderObjects.push_back(object);
                    mTotalModels.push_back(model);
                    mTotalTransformIds.push_back(transform->getId());

                    mTotalBoundingSpheres.push_back(computeWorldSpaceBoundingSphere(model, terrain->getChunkBounds(j)));
                }
            }
        }        
    }*/

    //assert(mTotalModels.size() == mTotalTransformIds.size());
    //assert(mTotalModels.size() == mTotalBoundingSpheres.size());

    assert(mCachedBoundingSpheres.size() == mFrustumVisible.size());

    mWorld->mBoundingSpheres = mCachedBoundingSpheres;
    mWorld->mFrustumVisible = mFrustumVisible;
    mWorld->mBoundingVolume = mCachedBoundingVolume;
}

void RenderSystem::buildRenderQueue()
{
    // mRenderQueue.clear();

    // for (size_t i = 0; i < mTotalRenderObjects.size(); i++)
    //{
    //     // for now dont sort
    //     uint64_t key = i;

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
    // std::sort(mRenderQueue.begin(), mRenderQueue.end(),
    //          [=](std::pair<uint64_t, int> &a, std::pair<uint64_t, int> &b) { return a.first > b.first; });
}


// should work only if matrix is calculated as M = T * R * S
glm::vec3 extractScale2(const glm::mat4 &m)
{
    glm::vec3 scale;

    scale.x = glm::length2(glm::vec3(m[0][0], m[0][1], m[0][2]));
    scale.y = glm::length2(glm::vec3(m[1][0], m[1][1], m[1][2]));
    scale.z = glm::length2(glm::vec3(m[2][0], m[2][1], m[2][2]));

    return scale;
}

float extractLargestScale(const glm::mat4 &m)
{
    glm::vec3 scale2 = extractScale2(m);

    return glm::sqrt(glm::max(scale2.x, glm::max(scale2.y, scale2.z)));
}

Sphere RenderSystem::computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere)
{
    Sphere boundingSphere;
    boundingSphere.mCentre = glm::vec3(model * glm::vec4(sphere.mCentre, 1.0f));
    boundingSphere.mRadius = sphere.mRadius * extractLargestScale(model);

    return boundingSphere;
}