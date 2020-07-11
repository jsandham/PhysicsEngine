#include <fstream>
#include <sstream>

#include "../include/EditorFileIO.h"

#include "core/WriteInternalToJson.h"
#include "core/WriteToJson.h"
#include "core/Log.h"
#include "core/Entity.h"

#include "components/MeshRenderer.h"
#include "components/Light.h"
#include "components/BoxCollider.h"
#include "components/SphereCollider.h"

#include "json/json.hpp"

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

bool PhysicsEditor::writeAssetToBinary(std::string filePath, std::string fileExtension, Guid id, std::string outFilePath)
{
	std::string infoMessage = "Writing binary version of asset " + filePath + " to library\n";
	Log::info(&infoMessage[0]);

	// load data from asset
	std::vector<char> data;
	int assetType = -1;
	if (fileExtension == "shader") {
		assetType = AssetType<Shader>::type;
		Shader shader;
		shader.load(filePath);
		data = shader.serialize(id);
	}
	else if (fileExtension == "png") {
		assetType = AssetType<Texture2D>::type;
		Texture2D texture;
		texture.load(filePath);
		data = texture.serialize(id);
	}
	else if (fileExtension == "obj") {
		assetType = AssetType<Mesh>::type;
		Mesh mesh;
		mesh.load(filePath);
		data = mesh.serialize(id);
	}
	else if (fileExtension == "material") {
		assetType = AssetType<Material>::type;
		Material material;
		material.load(filePath);
		data = material.serialize(id);
	}

	// write data to binary version of asset in library 
	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		AssetHeader header = {};
		header.mSignature = 0x9a9e9b4153534554;
		header.mType = assetType;
		header.mSize = data.size();
		header.mMajor = 0;
		header.mMinor = 1;
		header.mAssetId = id;

		outFile.write((char*)&header, sizeof(header));
		outFile.write(&data[0], data.size());

		outFile.close();
	}
	else {
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
	else {
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

	uint32_t entityCount = 0;
	uint32_t componentCount = 0;
	uint32_t systemCount = 0;
	for (it = objects.begin(); it != objects.end(); it++) {
		std::string type = it->second["type"].ToString();

		if (type == "Entity") {
			entities[it->first] = it->second;
			entityCount++;
		}
		else if (type == "Transform") {
			transforms[it->first] = it->second;
			componentCount++;
		}
		else if (type == "Camera") {
			cameras[it->first] = it->second;
			componentCount++
		}
		else if (type == "MeshRenderer") {
			meshRenderers[it->first] = it->second;
			componentCount++
		}
		else if (type == "Light") {
			lights[it->first] = it->second;
			componentCount++;
		}
		else if (type == "BoxCollider") {
			boxColliders[it->first] = it->second;
			componentCount++;
		}
		else if(type == "SphereCollider") {
			sphereColliders[it->first] = it->second;
			componentCount++;
		}
	}

	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		SceneHeader header = {};
		header.mSignature = 0x9a9e9b5343454e45;
		header.mEntityCount = entityCount;
		header.mComponentCount = componentCount;
		header.mSystemCount = systemCount;
		header.mMajor = 0;
		header.mMinor = 1;
		header.mSceneId = id;

		std::vector<EntityHeader> entityHeaders(entityCount);
		std::vector<ComponentHeader> componentHeaders(componentCount);
		std::vector<SystemHeader> systemHeaders(systemCount);

		std::vector<TransformHeader> transformHeaders(transforms.length());
		std::vector<CameraHeader> cameraHeaders(cameras.length());
		std::vector<MeshRendererHeader> meshRendererHeaders(meshRenderers.length());
		std::vector<LightHeader> lightHeaders(lights.length());
		std::vector<BoxColliderHeader> boxColliderHeaders(boxColliders.length());
		std::vector<SphereColliderHeader> sphereColliderHeaders(sphereColliders.length());

		header.mSize = sizeof(SceneHeader) +
					sizeof(EntityHeader) * entityHeaders.size() +
					sizeof(ComponentHeader) * componentHeaders.size() +
					sizeof(SystemHeader) * systemHeaders.size() +
					sizeof(TransformHeader) * transformHeaders.size() +
					sizeof(CameraHeader) * cameraHeaders.size() +
					sizeof(MeshRendererHeader) * meshRendererHeaders.size() +
					sizeof(LightHeader) * lightHeaders.size() +
					sizeof(BoxColliderHeader) * boxColliderHeaders.size() +
					sizeof(SphereColliderHeader) * sphereColliderHeaders.size();

		// serialize entities
		if (!entities.IsNull()) {
			int index = 0;
			json::JSON::JSONWrapper<map<string, JSON>> entityObjects = entities.ObjectRange();
			for (it = entityObjects.begin(); it != entityObjects.end(); it++) {
				
				entityHeaders[index].mEntityId = Guid(it->first);
				entityHeaders[index].mDoNotDestroy = false;
				entityHeaders[index].mType = EntityType<Entity>::type;

				index++;
			}
		}

		// serialize transforms
		if (!transforms.IsNull()) {
			int index = 0;
			json::JSON::JSONWrapper<map<string, JSON>> transformObjects = transforms.ObjectRange();
			for (it = transformObjects.begin(); it != transformObjects.end(); it++) {
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
			}
		}

		// serialize camera
		if (!cameras.IsNull()) {
			int index = 0;
			json::JSON::JSONWrapper<map<string, JSON>> cameraObjects = cameras.ObjectRange();
			for (it = cameraObjects.begin(); it != cameraObjects.end(); it++) {
				cameraHeaders[index].mComponentId = Guid(it->first);
				cameraHeaders[index].mEntityId = Guid(it->second["entity"].ToString());

				cameraHeaders[index].mTargetTextureId = Guid(it->second["targetTextureId"].ToString());

				cameraHeaders[index].mBackgroundColor.r = (float)it->second["backgroundColor"][0].ToFloat();
				cameraHeaders[index].mBackgroundColor.g = (float)it->second["backgroundColor"][1].ToFloat();
				cameraHeaders[index].mBackgroundColor.b = (float)it->second["backgroundColor"][2].ToFloat();
				cameraHeaders[index].mBackgroundColor.a = (float)it->second["backgroundColor"][3].ToFloat();

				cameraHeaders[index].mX = (int)it->second["x"].ToInt();
				cameraHeaders[index].mY = (int)it->second["y"].ToInt();
				cameraHeaders[index].mWidth = (int)it->second["width"].ToInt();
				cameraHeaders[index].mHeight = (int)it->second["height"].ToInt();

				cameraHeaders[index].mFov = (float)it->second["fov"].ToFloat();
				cameraHeaders[index].mNearPlane = (float)it->second["near"].ToFloat();
				cameraHeaders[index].mFarPlane = (float)it->second["far"].ToFloat();
			}
		}

		// serialize mesh renderers
		if (!meshRenderers.IsNull()) {
			int index = 0;
			json::JSON::JSONWrapper<map<string, JSON>> meshRendererObjects = meshRenderers.ObjectRange();
			for (it = meshRendererObjects.begin(); it != meshRendererObjects.end(); it++) {
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				MeshRenderer meshRenderer;

				//meshRenderer.componentId = Guid(it->first);
				//meshRenderer.entityId = Guid(it->second["entity"].ToString());
				meshRenderer.setMesh(Guid(it->second["mesh"].ToString()));

				if (it->second.hasKey("material")) {
					meshRenderer.mMaterialCount = 1;
					meshRenderer.setMaterial(Guid(it->second["material"].ToString()), 0);
					//meshRenderer.mMaterialIds[0] = Guid(it->second["material"].ToString());
					for (int j = 1; j < 8; j++) {
						meshRenderer.setMaterial(Guid::INVALID, j);
						//meshRenderer.mMaterialIds[j] = Guid::INVALID;
					}
				}
				else if (it->second.hasKey("materials")) {
					int materialCount = it->second["materials"].length();
					if (materialCount > 8) {
						Log::error("Currently only support at most 8 materials");
						return false;
					}

					meshRenderer.mMaterialCount = materialCount;

					for (int j = 0; j < materialCount; j++) {
						meshRenderer.setMaterial(Guid(it->second["materials"][j].ToString()), j);
						//meshRenderer.mMaterialIds[j] = Guid(it->second["materials"][j].ToString());
					}

					for (int j = materialCount; j < 8; j++) {
						meshRenderer.setMaterial(Guid::INVALID, j);
						//meshRenderer.mMaterialIds[j] = Guid::INVALID;
					}
				}

				meshRenderer.mIsStatic = it->second["isStatic"].ToBool();
				meshRenderer.mEnabled = it->second["enabled"].ToBool();

				std::vector<char> data = meshRenderer.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 3;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize lights
		if (!lights.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> lightObjects = lights.ObjectRange();
			for (it = lightObjects.begin(); it != lightObjects.end(); it++) {
				Light light;

				light.mAmbient.x = (float)it->second["ambient"][0].ToFloat();
				light.mAmbient.y = (float)it->second["ambient"][1].ToFloat();
				light.mAmbient.z = (float)it->second["ambient"][2].ToFloat();

				light.mDiffuse.x = (float)it->second["diffuse"][0].ToFloat();
				light.mDiffuse.y = (float)it->second["diffuse"][1].ToFloat();
				light.mDiffuse.z = (float)it->second["diffuse"][2].ToFloat();

				light.mSpecular.x = (float)it->second["specular"][0].ToFloat();
				light.mSpecular.y = (float)it->second["specular"][1].ToFloat();
				light.mSpecular.z = (float)it->second["specular"][2].ToFloat();

				light.mConstant = (float)it->second["constant"].ToFloat();
				light.mLinear = (float)it->second["linear"].ToFloat();
				light.mQuadratic = (float)it->second["quadratic"].ToFloat();
				light.mCutOff = (float)it->second["cutOff"].ToFloat();
				light.mOuterCutOff = (float)it->second["outerCutOff"].ToFloat();

				light.mLightType = static_cast<LightType>((int)it->second["lightType"].ToInt());
				light.mShadowType = static_cast<ShadowType>((int)it->second["shadowType"].ToInt());

				std::cout << "Light type: " << (int)light.mLightType << " shadow type: " << (int)light.mShadowType << std::endl;

				std::vector<char> data = light.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 5;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}
		if (!boxColliders.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> boxColliderObjects = boxColliders.ObjectRange();
			for (it = boxColliderObjects.begin(); it != boxColliderObjects.end(); it++) {
				BoxCollider collider;

				/*collider.componentId = Guid(it->first);
				collider.entityId = Guid(it->second["entity"].ToString());*/

				collider.mAABB.mCentre.x = (float)it->second["centre"][0].ToFloat();
				collider.mAABB.mCentre.y = (float)it->second["centre"][1].ToFloat();
				collider.mAABB.mCentre.z = (float)it->second["centre"][2].ToFloat();

				collider.mAABB.mSize.x = (float)it->second["size"][0].ToFloat();
				collider.mAABB.mSize.y = (float)it->second["size"][1].ToFloat();
				collider.mAABB.mSize.z = (float)it->second["size"][2].ToFloat();

				std::vector<char> data = collider.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 8;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}

		}
		if (!sphereColliders.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> sphereColliderObjects = sphereColliders.ObjectRange();
			for (it = sphereColliderObjects.begin(); it != sphereColliderObjects.end(); it++) {
				SphereCollider collider;

				//collider.componentId = Guid(it->first);
				//collider.entityId = Guid(it->second["entity"].ToString());

				collider.mSphere.mCentre.x = (float)it->second["centre"][0].ToFloat();
				collider.mSphere.mCentre.y = (float)it->second["centre"][1].ToFloat();
				collider.mSphere.mCentre.z = (float)it->second["centre"][2].ToFloat();

				collider.mSphere.mRadius = (float)it->second["radius"].ToFloat();

				std::vector<char> data = collider.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 9;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());

			}

		}

		outFile.write((char*)&header, sizeof(header));

		outFile.close();
	}
	else {
		std::string errorMessage = "Could not open file " + outFilePath + " for writing\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	return true;
}

bool PhysicsEditor::writeAssetToJson(PhysicsEngine::World* world, std::string outFilePath, PhysicsEngine::Guid assetId, int type)
{
	std::ofstream file;

	file.open(outFilePath, std::ios::out);

	if (!file.is_open()) {
		std::string message = "Could not write asset to file path " + outFilePath + "\n";
		PhysicsEngine::Log::error(message.c_str());
		return false;
	}

	json::JSON& assetObj = json::Object();

	PhysicsEngine::writeInternalAssetToJson(assetObj, world, assetId, type);

	file << assetObj;
	file << "\n";
	file.close();

	return true;
}

bool PhysicsEditor::writeSceneToJson(PhysicsEngine::World* world, std::string outFilePath, std::set<PhysicsEngine::Guid> editorOnlyEntityIds)
{
	std::ofstream file;

	file.open(outFilePath, std::ios::out);

	if (!file.is_open()) {
		std::string message = "Could not write world to file path " + outFilePath + "\n";
		PhysicsEngine::Log::error(message.c_str());
		return false;
	}

	json::JSON& sceneObj = json::Object();

	for (int i = 0; i < world->getNumberOfEntities(); i++) {
		Entity* entity = world->getEntityByIndex(i);

		// skip editor only entities
		std::set<PhysicsEngine::Guid>::iterator it = editorOnlyEntityIds.find(entity->getId());
		if (it != editorOnlyEntityIds.end()) {
			continue;
		}

		PhysicsEngine::writeInternalEntityToJson(sceneObj, world, entity->getId());

		std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);
		for (size_t j = 0; j < componentsOnEntity.size(); j++) {
			Guid componentId = componentsOnEntity[j].first;
			int componentType = componentsOnEntity[j].second;

			if (componentType < 20) {
				PhysicsEngine::writeInternalComponentToJson(sceneObj, world, entity->getId(), componentId, componentType);
			}
			else {
				PhysicsEngine::writeComponentToJson(sceneObj, world, entity->getId(), componentId, componentType);
			}
		}
	}

	for (int i = 0; i < world->getNumberOfSystems(); i++) {
		System* system = world->getSystemByIndex(i);

		Guid systemId = system->getId();
		int systemType = world->getTypeOf(system->getId());
		int systemOrder = system->getOrder();

		if (systemType < 20) {
			PhysicsEngine::writeInternalSystemToJson(sceneObj, world, systemId, systemType, systemOrder);
		}
		else {
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

	if (metaFile.is_open()) {
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

	if (metaFile.is_open()) {
		std::ostringstream contents;
		contents << metaFile.rdbuf();

		metaFile.close();

		std::string jsonContentString = contents.str();
		json::JSON object = json::JSON::Load(contents.str());

		return object["id"].ToString();
	}
	else {
		std::string errorMessage = "An error occured when trying to open meta file: " + metaFilePath + "\n";
		PhysicsEngine::Log::error(&errorMessage[0]);
		return PhysicsEngine::Guid::INVALID;
	}
}