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

		/*if (AssetLoader::load(filePath, shader)) {
			shader.assetId = id.toString();
			data = shader.serialize();
		}*/
	}
	else if (fileExtension == "png") {
		assetType = AssetType<Texture2D>::type;
		Texture2D texture;
		texture.load(filePath);
		data = texture.serialize(id);

		/*if (AssetLoader::load(filePath, texture)) {
			texture.assetId = id.toString();
			data = texture.serialize();
		}*/
	}
	else if (fileExtension == "obj") {
		assetType = AssetType<Mesh>::type;
		Mesh mesh;

		mesh.load(filePath);
		data = mesh.serialize(id);
		/*if (AssetLoader::load(filePath, mesh)) {
			mesh.assetId = id.toString();
			data = mesh.serialize();
		}*/
	}
	else if (fileExtension == "material") {
		assetType = AssetType<Material>::type;
		Material material;
		material.load(filePath);
		data = material.serialize(id);

		/*if (AssetLoader::load(filePath, material)) {
			material.assetId = id.toString();
			data = material.serialize();
		}*/
	}

	// write data to binary version of asset in library 
	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		AssetHeader header = {};

		outFile.write((char*)&header, sizeof(header));

		char classification = 'a';
		size_t size = data.size();

		outFile.write(&classification, 1);
		outFile.write((char*)&assetType, sizeof(int));
		outFile.write((char*)&size, sizeof(size_t));
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

	for (it = objects.begin(); it != objects.end(); it++) {
		std::string type = it->second["type"].ToString();

		if (type == "Entity") {
			entities[it->first] = it->second;
		}
		else if (type == "Transform") {
			transforms[it->first] = it->second;
		}
		else if (type == "Camera") {
			cameras[it->first] = it->second;
		}
		else if (type == "MeshRenderer") {
			meshRenderers[it->first] = it->second;
		}
		else if (type == "Light") {
			lights[it->first] = it->second;
		}
		else if (type == "BoxCollider") {
			boxColliders[it->first] = it->second;
		}
		else if(type == "SphereCollider") {
			sphereColliders[it->first] = it->second;
		}
	}

	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		SceneHeader header = {};

		outFile.write((char*)&header, sizeof(header));

		// serialize entities
		if (!entities.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> entityObjects = entities.ObjectRange();
			for (it = entityObjects.begin(); it != entityObjects.end(); it++) {
				Entity entity;
				//entity.entityId = Guid(it->first);
				entity.mDoNotDestroy = false;

				std::vector<char> data = entity.serialize(Guid(it->first));

				char classification = 'e';
				int type = 0;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize transforms
		if (!transforms.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> transformObjects = transforms.ObjectRange();
			for (it = transformObjects.begin(); it != transformObjects.end(); it++) {
				Transform transform;

				//transform.componentId = Guid(it->first);
				transform.mParentId = Guid(it->second["parent"].ToString());
				//transform.entityId = Guid(it->second["entity"].ToString());

				transform.mPosition.x = (float)it->second["position"][0].ToFloat();
				transform.mPosition.y = (float)it->second["position"][1].ToFloat();
				transform.mPosition.z = (float)it->second["position"][2].ToFloat();

				transform.mRotation.x = (float)it->second["rotation"][0].ToFloat();
				transform.mRotation.y = (float)it->second["rotation"][1].ToFloat();
				transform.mRotation.z = (float)it->second["rotation"][2].ToFloat();
				transform.mRotation.w = (float)it->second["rotation"][3].ToFloat();

				transform.mScale.x = (float)it->second["scale"][0].ToFloat();
				transform.mScale.y = (float)it->second["scale"][1].ToFloat();
				transform.mScale.z = (float)it->second["scale"][2].ToFloat();

				std::vector<char> data = transform.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 0;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize camera
		if (!cameras.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> cameraObjects = cameras.ObjectRange();
			for (it = cameraObjects.begin(); it != cameraObjects.end(); it++) {
				Camera camera;

				//camera.componentId = Guid(it->first);
				//camera.entityId = Guid(it->second["entity"].ToString());
				camera.mTargetTextureId = Guid(it->second["targetTextureId"].ToString());

				camera.mPosition.x = (float)it->second["position"][0].ToFloat();
				camera.mPosition.y = (float)it->second["position"][1].ToFloat();
				camera.mPosition.z = (float)it->second["position"][2].ToFloat();

				camera.mFront.x = (float)it->second["front"][0].ToFloat();
				camera.mFront.y = (float)it->second["front"][1].ToFloat();
				camera.mFront.z = (float)it->second["front"][2].ToFloat();

				camera.mUp.x = (float)it->second["up"][0].ToFloat();
				camera.mUp.y = (float)it->second["up"][1].ToFloat();
				camera.mUp.z = (float)it->second["up"][2].ToFloat();

				camera.mBackgroundColor.x = (float)it->second["backgroundColor"][0].ToFloat();
				camera.mBackgroundColor.y = (float)it->second["backgroundColor"][1].ToFloat();
				camera.mBackgroundColor.z = (float)it->second["backgroundColor"][2].ToFloat();
				camera.mBackgroundColor.w = (float)it->second["backgroundColor"][3].ToFloat();

				camera.mViewport.mX = (int)it->second["x"].ToInt();
				camera.mViewport.mY = (int)it->second["y"].ToInt();
				camera.mViewport.mWidth = (int)it->second["width"].ToInt();
				camera.mViewport.mHeight = (int)it->second["height"].ToInt();

				camera.mFrustum.mFov = (float)it->second["fov"].ToFloat();
				camera.mFrustum.mNearPlane = (float)it->second["near"].ToFloat();
				camera.mFrustum.mFarPlane = (float)it->second["far"].ToFloat();

				std::vector<char> data = camera.serialize(Guid(it->first), Guid(it->second["entity"].ToString()));

				char classification = 'c';
				int type = 2;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize mesh renderers
		if (!meshRenderers.IsNull()) {
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

				//light.componentId = Guid(it->first);
				//light.entityId = Guid(it->second["entity"].ToString());

				light.mPosition.x = (float)it->second["position"][0].ToFloat();
				light.mPosition.y = (float)it->second["position"][1].ToFloat();
				light.mPosition.z = (float)it->second["position"][2].ToFloat();

				light.mDirection.x = (float)it->second["direction"][0].ToFloat();
				light.mDirection.y = (float)it->second["direction"][1].ToFloat();
				light.mDirection.z = (float)it->second["direction"][2].ToFloat();

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

				collider.mBounds.mCentre.x = (float)it->second["centre"][0].ToFloat();
				collider.mBounds.mCentre.y = (float)it->second["centre"][1].ToFloat();
				collider.mBounds.mCentre.z = (float)it->second["centre"][2].ToFloat();

				collider.mBounds.mSize.x = (float)it->second["size"][0].ToFloat();
				collider.mBounds.mSize.y = (float)it->second["size"][1].ToFloat();
				collider.mBounds.mSize.z = (float)it->second["size"][2].ToFloat();

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

bool PhysicsEditor::writeSceneToJson(PhysicsEngine::World* world, std::string outFilePath)
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

		if (entity->getId() == Guid("11111111-1111-1111-1111-111111111111")) {
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

















	//std::ofstream file;

	//std::string test = "Writing world to file " + outFilePath + "\n";
	//PhysicsEngine::Log::info(test.c_str());

	//file.open(outFilePath, std::ios::out);

	//if (!file.is_open()) {
	//	std::string message = "Could not write world to file path " + outFilePath + "\n";
	//	PhysicsEngine::Log::error(message.c_str());
	//	return false;
	//}

	//json::JSON obj;

	//for (int i = 0; i < world->getNumberOfEntities(); i++) {
	//	Entity* entity = world->getEntityByIndex(i);

	//	if (entity->entityId == Guid("11111111-1111-1111-1111-111111111111")) {
	//		continue;
	//	}

	//	std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);

	//	Guid entityId = entity->entityId;

	//	// write entity to json
	//	obj[entityId.toString()] = json::Object();
	//	obj[entityId.toString()]["type"] = "Entity";
	//	for (size_t j = 0; j < componentsOnEntity.size(); j++) {
	//		obj[entityId.toString()]["components"].append(componentsOnEntity[j].first.toString());
	//	}

	//	// write each component on entity to json
	//	for (size_t j = 0; j < componentsOnEntity.size(); j++) {
	//		Guid componentId = componentsOnEntity[j].first;
	//		int componentType = componentsOnEntity[j].second;

	//		//json::JSON componentObj = json::Object();
	//		if (componentType == 0) {
	//			Transform* transform = world->getComponent<Transform>(entityId);
	//			
	//			obj[componentId.toString()]["type"] = "Transform";
	//			obj[componentId.toString()]["parent"] = transform->parentId.toString();
	//			obj[componentId.toString()]["entity"] = entityId.toString();
	//			obj[componentId.toString()]["position"].append(transform->position.x, transform->position.y, transform->position.z);
	//			obj[componentId.toString()]["rotation"].append(transform->rotation.x, transform->rotation.y, transform->rotation.z, transform->rotation.w);
	//			obj[componentId.toString()]["scale"].append(transform->scale.x, transform->scale.y, transform->scale.z);
	//		}
	//		else if (componentType == 1) {
	//			Rigidbody* rigidbody = world->getComponent<Rigidbody>(entityId);
	//		}
	//		else if (componentType == 2) {
	//			Camera* camera = world->getComponent<Camera>(entityId);

	//			obj[componentId.toString()]["type"] = "Camera";
	//			obj[componentId.toString()]["entity"] = entityId.toString();
	//			obj[componentId.toString()]["targetTextureId"] = camera->targetTextureId.toString();
	//			obj[componentId.toString()]["position"].append(camera->position.x, camera->position.y, camera->position.z);
	//			obj[componentId.toString()]["front"].append(camera->front.x, camera->front.y, camera->front.z);
	//			obj[componentId.toString()]["up"].append(camera->up.x, camera->up.y, camera->up.z);
	//			obj[componentId.toString()]["backgroundColor"].append(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
	//			obj[componentId.toString()]["x"] = camera->viewport.x;
	//			obj[componentId.toString()]["y"] = camera->viewport.y;
	//			obj[componentId.toString()]["width"] = camera->viewport.width;
	//			obj[componentId.toString()]["height"] = camera->viewport.height;
	//			obj[componentId.toString()]["fov"] = camera->frustum.fov;
	//			obj[componentId.toString()]["near"] = camera->frustum.nearPlane;
	//			obj[componentId.toString()]["far"] = camera->frustum.farPlane;
	//		}
	//		else if (componentType == 3) {
	//			MeshRenderer* meshRenderer = world->getComponent<MeshRenderer>(entityId);

	//			obj[componentId.toString()]["type"] = "MeshRenderer";
	//			obj[componentId.toString()]["entity"] = entityId.toString();
	//			obj[componentId.toString()]["mesh"] = meshRenderer->meshId.toString();

	//			int materialCount = meshRenderer->materialCount;

	//			std::string label = "material";
	//			if (materialCount > 1) {
	//				label = "materials";
	//			}

	//			std::string value = "";
	//			if (materialCount == 0) {
	//				value = Guid::INVALID.toString();
	//			}
	//			else if (materialCount == 1) {
	//				value = meshRenderer->materialIds[0].toString();
	//			}
	//			else { // dont think this is right. I think I need to do something like obj[componentId.toString()][label].append...
	//				value += "[";
	//				for (int m = 0; m < materialCount; m++) {
	//					value += meshRenderer->materialIds[m].toString();
	//					if (m != materialCount - 1) {
	//						value += ",";
	//					}
	//				}
	//				value += "]";
	//			}
	//		
	//			obj[componentId.toString()][label] = value;
	//			obj[componentId.toString()]["isStatic"] = meshRenderer->isStatic;
	//		}
	//		else if (componentType == 4) {
	//			LineRenderer* lineRenderer = world->getComponent<LineRenderer>(entityId);
	//		}
	//		else if (componentType == 5) {
	//			Light* light = world->getComponent<Light>(entityId);

	//			obj[componentId.toString()]["type"] = "Light";
	//			obj[componentId.toString()]["entity"] = entityId.toString();
	//			obj[componentId.toString()]["position"].append(light->position.x, light->position.y, light->position.z);
	//			obj[componentId.toString()]["direction"].append(light->direction.x, light->direction.y, light->direction.z);
	//			obj[componentId.toString()]["ambient"].append(light->ambient.x, light->ambient.y, light->ambient.z);
	//			obj[componentId.toString()]["diffuse"].append(light->diffuse.x, light->diffuse.y, light->diffuse.z);
	//			obj[componentId.toString()]["specular"].append(light->specular.x, light->specular.y, light->specular.z);
	//			obj[componentId.toString()]["constant"] = light->constant;
	//			obj[componentId.toString()]["linear"] = light->linear;
	//			obj[componentId.toString()]["quadratic"] = light->quadratic;
	//			obj[componentId.toString()]["cutOff"] = light->cutOff;
	//			obj[componentId.toString()]["outerCutOff"] = light->outerCutOff;
	//			obj[componentId.toString()]["lightType"] = static_cast<int>(light->lightType);
	//			obj[componentId.toString()]["shadowType"] = static_cast<int>(light->shadowType);
	//		}
	//		else if (componentType == 8) {
	//			BoxCollider* collider = world->getComponent<BoxCollider>(entityId);

	//			obj[componentId.toString()]["type"] = "SphereCollider";
	//			obj[componentId.toString()]["entity"] = entityId.toString();

	//			obj[componentId.toString()]["centre"].append(collider->bounds.centre.x, collider->bounds.centre.y, collider->bounds.centre.z);
	//			obj[componentId.toString()]["size"].append(collider->bounds.size.x, collider->bounds.size.y, collider->bounds.size.z);
	//		}
	//		else if (componentType == 9) {
	//			SphereCollider* collider = world->getComponent<SphereCollider>(entityId);

	//			obj[componentId.toString()]["type"] = "SphereCollider";
	//			obj[componentId.toString()]["entity"] = entityId.toString();

	//			obj[componentId.toString()]["centre"].append(collider->sphere.centre.x, collider->sphere.centre.y, collider->sphere.centre.z);
	//			obj[componentId.toString()]["radius"] = collider->sphere.radius;
	//		}
	//		else if (componentType == 10) {
	//			CapsuleCollider* capsuleCollider = world->getComponent<CapsuleCollider>(entityId);
	//		}
	//		else if (componentType == 15) {
	//			MeshCollider* meshCollider = world->getComponent<MeshCollider>(entityId);
	//		}
	//	}



		//// Create a new Array as a field of an Object.
		//obj["array"] = json::Array(true, "Two", 3, 4.0);
		//// Create a new Object as a field of another Object.
		//obj["obj"] = json::Object();
		//// Assign to one of the inner object's fields
		//obj["obj"]["inner"] = "Inside";

		//// We don't need to specify the type of the JSON object:
		//obj["new"]["some"]["deep"]["key"] = "Value";
		//obj["array2"].append(false, "three");

		//// We can also parse a string into a JSON object:
		//obj["parsed"] = JSON::Load("[ { \"Key\" : \"Value\" }, false ]");
	//}

	/*file << obj;
	file << "\n";

	file.close();

	return true;*/
}