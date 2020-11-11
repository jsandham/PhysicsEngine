#include <fstream>
#include <sstream>

#include "../include/EditorFileIO.h"

#include "core/Entity.h"
#include "core/Log.h"
#include "core/WorldSerialization.h"
#include "core/WriteInternalToJson.h"
#include "core/WriteToJson.h"

#include "components/BoxCollider.h"
#include "components/Light.h"
#include "components/MeshRenderer.h"
#include "components/SphereCollider.h"

#include "json/json.hpp"

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

bool PhysicsEditor::writeAssetToBinary(std::string filePath, std::string fileExtension, Guid id,
                                       std::string outFilePath)
{
    std::string infoMessage = "Writing binary version of asset " + filePath + " to library\n";
    Log::info(&infoMessage[0]);

    // load data from asset
    std::vector<char> data;
    int assetType = -1;
    if (fileExtension == "shader")
    {
        assetType = AssetType<Shader>::type;
        Shader shader;
        shader.load(filePath);
        data = shader.serialize(id);
    }
    else if (fileExtension == "png")
    {
        assetType = AssetType<Texture2D>::type;
        Texture2D texture;
        texture.load(filePath);
        data = texture.serialize(id);
    }
    else if (fileExtension == "obj")
    {
        assetType = AssetType<Mesh>::type;
        Mesh mesh;
        mesh.load(filePath);
        data = mesh.serialize(id);
    }
    else if (fileExtension == "material")
    {
        assetType = AssetType<Material>::type;
        Material material;
        material.load(filePath);
        data = material.serialize(id);
    }

    // write data to binary version of asset in library
    std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

    if (outFile.is_open())
    {
        AssetFileHeader header = {};
        header.mSignature = PhysicsEngine::ASSET_FILE_SIGNATURE;
        header.mType = static_cast<int32_t>(assetType);
        header.mSize = data.size();
        header.mMajor = 0;
        header.mMinor = 1;
        header.mAssetId = id;

        outFile.write((char *)&header, sizeof(header));
        outFile.write(&data[0], data.size());

        outFile.close();
    }
    else
    {
        std::string errorMessage = "Could not open file " + outFilePath + " for writing to library\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    return true;
}

bool PhysicsEditor::writeSceneToBinary(std::string filePath, Guid id, std::string outFilePath)
{
    std::string infoMessage = "Writing binary version of scene " + filePath + " to library\n";
    Log::info(&infoMessage[0]);

    std::fstream file;

    file.open(filePath);

    std::ostringstream contents;
    if (file.is_open())
    {
        contents << file.rdbuf();
        file.close();
    }
    else
    {
        std::string errorMessage = "Could not open scene " + filePath + " for writing to library\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    std::string contentString = contents.str();
    json::JSON jsonScene = json::JSON::Load(contentString);

    // parse loaded json file
    json::JSON entities;
    json::JSON transforms;
    json::JSON cameras;
    json::JSON meshRenderers;
    json::JSON lights;
    json::JSON boxColliders;
    json::JSON sphereColliders;

    json::JSON::JSONWrapper<map<string, JSON>> objects = jsonScene.ObjectRange();
    map<string, JSON>::iterator it;

    int32_t entityCount = 0;
    int32_t componentCount = 0;
    int32_t systemCount = 0;
    int32_t transformCount = 0;
    int32_t cameraCount = 0;
    int32_t meshRendererCount = 0;
    int32_t lightCount = 0;
    int32_t boxColliderCount = 0;
    int32_t sphereColliderCount = 0;
    for (it = objects.begin(); it != objects.end(); it++)
    {
        std::string type = it->second["type"].ToString();

        if (type == "Entity")
        {
            entities[it->first] = it->second;
            entityCount++;
        }
        else if (type == "Transform")
        {
            transforms[it->first] = it->second;
            componentCount++;
            transformCount++;
        }
        else if (type == "Camera")
        {
            cameras[it->first] = it->second;
            componentCount++;
            cameraCount++;
        }
        else if (type == "MeshRenderer")
        {
            meshRenderers[it->first] = it->second;
            componentCount++;
            meshRendererCount++;
        }
        else if (type == "Light")
        {
            lights[it->first] = it->second;
            componentCount++;
            lightCount++;
        }
        else if (type == "BoxCollider")
        {
            boxColliders[it->first] = it->second;
            componentCount++;
            boxColliderCount++;
        }
        else if (type == "SphereCollider")
        {
            sphereColliders[it->first] = it->second;
            componentCount++;
            sphereColliderCount++;
        }
    }

    SceneFileHeader header = {};
    header.mSignature = PhysicsEngine::SCENE_FILE_SIGNATURE;
    header.mEntityCount = entityCount;
    header.mComponentCount = componentCount;
    header.mSystemCount = systemCount;
    header.mMajor = 0;
    header.mMinor = 1;
    header.mSceneId = id;

    std::vector<ComponentInfoHeader> componentHeaders(componentCount);
    std::vector<SystemInfoHeader> systemHeaders(systemCount);

    std::vector<EntityHeader> entityHeaders(entityCount);
    std::vector<TransformHeader> transformHeaders(transformCount);
    std::vector<CameraHeader> cameraHeaders(cameraCount);
    std::vector<MeshRendererHeader> meshRendererHeaders(meshRendererCount);
    std::vector<LightHeader> lightHeaders(lightCount);
    std::vector<BoxColliderHeader> boxColliderHeaders(boxColliderCount);
    std::vector<SphereColliderHeader> sphereColliderHeaders(sphereColliderCount);

    header.mSize = sizeof(SceneFileHeader) + sizeof(ComponentInfoHeader) * componentHeaders.size() +
                   sizeof(SystemInfoHeader) * systemHeaders.size() + sizeof(EntityHeader) * entityHeaders.size() +
                   sizeof(TransformHeader) * transformHeaders.size() + sizeof(CameraHeader) * cameraHeaders.size() +
                   sizeof(MeshRendererHeader) * meshRendererHeaders.size() + sizeof(LightHeader) * lightHeaders.size() +
                   sizeof(BoxColliderHeader) * boxColliderHeaders.size() +
                   sizeof(SphereColliderHeader) * sphereColliderHeaders.size();

    // serialize entities
    if (!entities.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> entityObjects = entities.ObjectRange();
        for (it = entityObjects.begin(); it != entityObjects.end(); it++)
        {

            entityHeaders[index].mEntityId = Guid(it->first);

            std::string name = it->second["name"].ToString();
            std::size_t len = std::min(size_t(64 - 1), name.size());
            memcpy(&(entityHeaders[index].mEntityName[0]), &name[0], len);
            entityHeaders[index].mEntityName[len] = '\0';

            entityHeaders[index].mDoNotDestroy = static_cast<uint8_t>(false);

            index++;
        }
    }

    int componentIndex = 0;

    // serialize transforms
    if (!transforms.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> transformObjects = transforms.ObjectRange();
        for (it = transformObjects.begin(); it != transformObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<Transform>::type);
            componentHeaders[componentIndex].mSize = sizeof(TransformHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            transformHeaders[index].mComponentId = Guid(it->first);
            transformHeaders[index].mEntityId = Guid(it->second["entity"].ToString());
            transformHeaders[index].mParentId = Guid(it->second["parent"].ToString());

            transformHeaders[index].mPosition.x = (float)it->second["position"][0].ToFloat();
            transformHeaders[index].mPosition.y = (float)it->second["position"][1].ToFloat();
            transformHeaders[index].mPosition.z = (float)it->second["position"][2].ToFloat();

            transformHeaders[index].mRotation.x = (float)it->second["rotation"][0].ToFloat();
            transformHeaders[index].mRotation.y = (float)it->second["rotation"][1].ToFloat();
            transformHeaders[index].mRotation.z = (float)it->second["rotation"][2].ToFloat();
            transformHeaders[index].mRotation.w = (float)it->second["rotation"][3].ToFloat();

            transformHeaders[index].mScale.x = (float)it->second["scale"][0].ToFloat();
            transformHeaders[index].mScale.y = (float)it->second["scale"][1].ToFloat();
            transformHeaders[index].mScale.z = (float)it->second["scale"][2].ToFloat();

            index++;
            componentIndex++;
        }
    }

    // serialize camera
    if (!cameras.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> cameraObjects = cameras.ObjectRange();
        for (it = cameraObjects.begin(); it != cameraObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<Camera>::type);
            componentHeaders[componentIndex].mSize = sizeof(CameraHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            cameraHeaders[index].mComponentId = Guid(it->first);
            cameraHeaders[index].mEntityId = Guid(it->second["entity"].ToString());

            cameraHeaders[index].mTargetTextureId = Guid(it->second["targetTextureId"].ToString());

            cameraHeaders[index].mBackgroundColor.r = (float)it->second["backgroundColor"][0].ToFloat();
            cameraHeaders[index].mBackgroundColor.g = (float)it->second["backgroundColor"][1].ToFloat();
            cameraHeaders[index].mBackgroundColor.b = (float)it->second["backgroundColor"][2].ToFloat();
            cameraHeaders[index].mBackgroundColor.a = (float)it->second["backgroundColor"][3].ToFloat();

            cameraHeaders[index].mX = static_cast<int32_t>(it->second["x"].ToInt());
            cameraHeaders[index].mY = static_cast<int32_t>(it->second["y"].ToInt());
            cameraHeaders[index].mWidth = static_cast<int32_t>(it->second["width"].ToInt());
            cameraHeaders[index].mHeight = static_cast<int32_t>(it->second["height"].ToInt());

            cameraHeaders[index].mFov = (float)it->second["fov"].ToFloat();
            cameraHeaders[index].mNearPlane = (float)it->second["near"].ToFloat();
            cameraHeaders[index].mFarPlane = (float)it->second["far"].ToFloat();

            index++;
            componentIndex++;
        }
    }

    // serialize mesh renderers
    if (!meshRenderers.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> meshRendererObjects = meshRenderers.ObjectRange();
        for (it = meshRendererObjects.begin(); it != meshRendererObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<MeshRenderer>::type);
            componentHeaders[componentIndex].mSize = sizeof(MeshRendererHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            meshRendererHeaders[index].mComponentId = Guid(it->first);
            meshRendererHeaders[index].mEntityId = Guid(it->second["entity"].ToString());
            meshRendererHeaders[index].mMeshId = Guid(it->second["mesh"].ToString());

            if (it->second.hasKey("material"))
            {
                meshRendererHeaders[index].mMaterialCount = 1;
                meshRendererHeaders[index].mMaterialIds[0] = Guid(it->second["material"].ToString());

                for (int j = 1; j < 8; j++)
                {
                    meshRendererHeaders[index].mMaterialIds[j] = Guid::INVALID;
                }
            }
            else if (it->second.hasKey("materials"))
            {
                int materialCount = it->second["materials"].length();
                if (materialCount > 8)
                {
                    Log::error("Currently only support at most 8 materials");
                    return false;
                }

                meshRendererHeaders[index].mMaterialCount = static_cast<int32_t>(materialCount);

                for (int j = 0; j < materialCount; j++)
                {
                    meshRendererHeaders[index].mMaterialIds[j] = Guid(it->second["materials"][j].ToString());
                }

                for (int j = materialCount; j < 8; j++)
                {
                    meshRendererHeaders[index].mMaterialIds[j] = Guid::INVALID;
                }
            }

            meshRendererHeaders[index].mIsStatic = static_cast<uint8_t>(it->second["isStatic"].ToBool());
            meshRendererHeaders[index].mEnabled = static_cast<uint8_t>(it->second["enabled"].ToBool());

            index++;
            componentIndex++;
        }
    }

    // serialize lights
    if (!lights.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> lightObjects = lights.ObjectRange();
        for (it = lightObjects.begin(); it != lightObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<Light>::type);
            componentHeaders[componentIndex].mSize = sizeof(LightHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            lightHeaders[index].mComponentId = Guid(it->first);
            lightHeaders[index].mEntityId = Guid(it->second["entity"].ToString());

            lightHeaders[index].mColor.x = (float)it->second["color"][0].ToFloat();
            lightHeaders[index].mColor.y = (float)it->second["color"][1].ToFloat();
            lightHeaders[index].mColor.z = (float)it->second["color"][2].ToFloat();
            lightHeaders[index].mColor.w = (float)it->second["color"][3].ToFloat();

            lightHeaders[index].mIntensity = (float)it->second["intensity"].ToFloat();
            lightHeaders[index].mSpotAngle = (float)it->second["spotAngle"].ToFloat();
            lightHeaders[index].mInnerSpotAngle = (float)it->second["innerSpotAngle"].ToFloat();
            lightHeaders[index].mShadowNearPlane = (float)it->second["shadowNearPlane"].ToFloat();
            lightHeaders[index].mShadowFarPlane = (float)it->second["shadowFarPlane"].ToFloat();
            lightHeaders[index].mShadowAngle = (float)it->second["shadowAngle"].ToFloat();
            lightHeaders[index].mShadowRadius = (float)it->second["shadowRadius"].ToFloat();
            lightHeaders[index].mShadowStrength = (float)it->second["shadowStrength"].ToFloat();

            lightHeaders[index].mLightType = static_cast<uint8_t>((int)it->second["lightType"].ToInt());
            lightHeaders[index].mShadowType = static_cast<uint8_t>((int)it->second["shadowType"].ToInt());
            lightHeaders[index].mShadowMapResolution =
                static_cast<uint16_t>((int)it->second["shadowMapResolution"].ToInt());

            index++;
            componentIndex++;
        }
    }
    if (!boxColliders.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> boxColliderObjects = boxColliders.ObjectRange();
        for (it = boxColliderObjects.begin(); it != boxColliderObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<BoxCollider>::type);
            componentHeaders[componentIndex].mSize = sizeof(BoxColliderHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            boxColliderHeaders[index].mComponentId = Guid(it->first);
            boxColliderHeaders[index].mEntityId = Guid(it->second["entity"].ToString());

            boxColliderHeaders[index].mAABB.mCentre.x = (float)it->second["centre"][0].ToFloat();
            boxColliderHeaders[index].mAABB.mCentre.y = (float)it->second["centre"][1].ToFloat();
            boxColliderHeaders[index].mAABB.mCentre.z = (float)it->second["centre"][2].ToFloat();

            boxColliderHeaders[index].mAABB.mSize.x = (float)it->second["size"][0].ToFloat();
            boxColliderHeaders[index].mAABB.mSize.y = (float)it->second["size"][1].ToFloat();
            boxColliderHeaders[index].mAABB.mSize.z = (float)it->second["size"][2].ToFloat();

            index++;
            componentIndex++;
        }
    }
    if (!sphereColliders.IsNull())
    {
        int index = 0;
        json::JSON::JSONWrapper<map<string, JSON>> sphereColliderObjects = sphereColliders.ObjectRange();
        for (it = sphereColliderObjects.begin(); it != sphereColliderObjects.end(); it++)
        {
            componentHeaders[componentIndex].mComponentId = Guid(it->first);
            componentHeaders[componentIndex].mType = static_cast<int32_t>(ComponentType<SphereCollider>::type);
            componentHeaders[componentIndex].mSize = sizeof(SphereColliderHeader);
            componentHeaders[componentIndex].mStartPtr = 0;

            sphereColliderHeaders[index].mComponentId = Guid(it->first);
            sphereColliderHeaders[index].mEntityId = Guid(it->second["entity"].ToString());

            sphereColliderHeaders[index].mSphere.mCentre.x = (float)it->second["centre"][0].ToFloat();
            sphereColliderHeaders[index].mSphere.mCentre.y = (float)it->second["centre"][1].ToFloat();
            sphereColliderHeaders[index].mSphere.mCentre.z = (float)it->second["centre"][2].ToFloat();

            sphereColliderHeaders[index].mSphere.mRadius = (float)it->second["radius"].ToFloat();

            index++;
            componentIndex++;
        }
    }

    // determine start pointer offsets for components
    componentIndex = 0;
    /*size_t offset = sizeof(SceneFileHeader) +
                    sizeof(ComponentInfoHeader) * componentHeaders.size() +
                    sizeof(SystemInfoHeader) * systemHeaders.size() +
                    sizeof(EntityHeader)* entityHeaders.size();*/
    size_t offset = sizeof(ComponentInfoHeader) * componentHeaders.size() +
                    sizeof(SystemInfoHeader) * systemHeaders.size() + sizeof(EntityHeader) * entityHeaders.size();

    // transforms
    for (size_t i = 0; i < transformHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(transformHeaders[i]);
        componentIndex++;
    }

    // cameras
    for (size_t i = 0; i < cameraHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(cameraHeaders[i]);
        componentIndex++;
    }

    // mesh renderers
    for (size_t i = 0; i < meshRendererHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(meshRendererHeaders[i]);
        componentIndex++;
    }

    // lights
    for (size_t i = 0; i < lightHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(lightHeaders[i]);
        componentIndex++;
    }

    // box colliders
    for (size_t i = 0; i < boxColliderHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(boxColliderHeaders[i]);
        componentIndex++;
    }

    // sphere colliders
    for (size_t i = 0; i < sphereColliderHeaders.size(); i++)
    {
        componentHeaders[componentIndex].mStartPtr = offset;

        offset += sizeof(sphereColliderHeaders[i]);
        componentIndex++;
    }

    // write data out to binary scene file
    std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

    if (outFile.is_open())
    {
        // write scene file header
        outFile.write((char *)&header, sizeof(header));

        // component info
        for (size_t i = 0; i < componentHeaders.size(); i++)
        {
            outFile.write((char *)&componentHeaders[i], sizeof(componentHeaders[i]));

            size_t test = sizeof(componentHeaders[i]);
        }

        // system info
        for (size_t i = 0; i < systemHeaders.size(); i++)
        {
            outFile.write((char *)&systemHeaders[i], sizeof(systemHeaders[i]));
        }

        // entities
        for (size_t i = 0; i < entityHeaders.size(); i++)
        {
            outFile.write((char *)&entityHeaders[i], sizeof(entityHeaders[i]));
        }

        // transforms
        for (size_t i = 0; i < transformHeaders.size(); i++)
        {
            outFile.write((char *)&transformHeaders[i], sizeof(transformHeaders[i]));
        }

        // cameras
        for (size_t i = 0; i < cameraHeaders.size(); i++)
        {
            outFile.write((char *)&cameraHeaders[i], sizeof(cameraHeaders[i]));
        }

        // mesh renderers
        for (size_t i = 0; i < meshRendererHeaders.size(); i++)
        {
            outFile.write((char *)&meshRendererHeaders[i], sizeof(meshRendererHeaders[i]));
        }

        // lights
        for (size_t i = 0; i < lightHeaders.size(); i++)
        {
            outFile.write((char *)&lightHeaders[i], sizeof(lightHeaders[i]));
        }

        // box colliders
        for (size_t i = 0; i < boxColliderHeaders.size(); i++)
        {
            outFile.write((char *)&boxColliderHeaders[i], sizeof(boxColliderHeaders[i]));
        }

        // sphere colliders
        for (size_t i = 0; i < sphereColliderHeaders.size(); i++)
        {
            outFile.write((char *)&sphereColliderHeaders[i], sizeof(sphereColliderHeaders[i]));
        }

        outFile.close();
    }
    else
    {
        std::string errorMessage = "Could not open file " + outFilePath + " for writing\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    return true;
}

bool PhysicsEditor::writeAssetToJson(PhysicsEngine::World *world, std::string outFilePath, PhysicsEngine::Guid assetId,
                                     int type)
{
    std::ofstream file;

    file.open(outFilePath, std::ios::out);

    if (!file.is_open())
    {
        std::string message = "Could not write asset to file path " + outFilePath + "\n";
        PhysicsEngine::Log::error(message.c_str());
        return false;
    }

    json::JSON &assetObj = json::Object();

    PhysicsEngine::writeInternalAssetToJson(assetObj, world, assetId, type);

    file << assetObj;
    file << "\n";
    file.close();

    return true;
}

bool PhysicsEditor::writeSceneToJson(PhysicsEngine::World *world, std::string outFilePath,
                                     std::set<PhysicsEngine::Guid> editorOnlyEntityIds)
{
    std::ofstream file;

    file.open(outFilePath, std::ios::out);

    if (!file.is_open())
    {
        std::string message = "Could not write world to file path " + outFilePath + "\n";
        PhysicsEngine::Log::error(message.c_str());
        return false;
    }

    json::JSON &sceneObj = json::Object();

    for (int i = 0; i < world->getNumberOfEntities(); i++)
    {
        Entity *entity = world->getEntityByIndex(i);

        // skip editor only entities
        std::set<PhysicsEngine::Guid>::iterator it = editorOnlyEntityIds.find(entity->getId());
        if (it != editorOnlyEntityIds.end())
        {
            continue;
        }

        PhysicsEngine::writeInternalEntityToJson(sceneObj, world, entity->getId());

        std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);
        for (size_t j = 0; j < componentsOnEntity.size(); j++)
        {
            Guid componentId = componentsOnEntity[j].first;
            int componentType = componentsOnEntity[j].second;

            if (Component::isInternal(componentType))
            {
                PhysicsEngine::writeInternalComponentToJson(sceneObj, world, entity->getId(), componentId,
                                                            componentType);
            }
            else
            {
                PhysicsEngine::writeComponentToJson(sceneObj, world, entity->getId(), componentId, componentType);
            }
        }
    }

    for (int i = 0; i < world->getNumberOfUpdatingSystems(); i++)
    {
        System *system = world->getSystemByUpdateOrder(i);

        Guid systemId = system->getId();
        int systemType = world->getTypeOf(system->getId());
        int systemOrder = system->getOrder();

        if (System::isInternal(systemType))
        {
            PhysicsEngine::writeInternalSystemToJson(sceneObj, world, systemId, systemType, systemOrder);
        }
        else
        {
            PhysicsEngine::writeSystemToJson(sceneObj, world, systemId, systemType, systemOrder);
        }
    }

    file << sceneObj;
    file << "\n";
    file.close();

    return true;
}

bool PhysicsEditor::createMetaFile(std::string metaFilePath)
{
    std::fstream metaFile;

    metaFile.open(metaFilePath, std::fstream::out);

    if (metaFile.is_open())
    {
        metaFile << "{\n";
        metaFile << "\t\"id\" : \"" + PhysicsEngine::Guid::newGuid().toString() + "\"\n";
        metaFile << "}\n";
        metaFile.close();

        return true;
    }

    return false;
}

PhysicsEngine::Guid PhysicsEditor::findGuidFromMetaFilePath(std::string metaFilePath)
{
    // get guid from meta file
    std::fstream metaFile;
    metaFile.open(metaFilePath, std::fstream::in);

    if (metaFile.is_open())
    {
        std::ostringstream contents;
        contents << metaFile.rdbuf();

        metaFile.close();

        std::string jsonContentString = contents.str();
        json::JSON object = json::JSON::Load(contents.str());

        return object["id"].ToString();
    }
    else
    {
        std::string errorMessage = "An error occured when trying to open meta file: " + metaFilePath + "\n";
        PhysicsEngine::Log::error(&errorMessage[0]);
        return PhysicsEngine::Guid::INVALID;
    }
}