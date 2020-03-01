#include "../../include/core/WriteInternalToJson.h"

#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Texture3D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Material.h"
#include "../../include/core/Mesh.h"
#include "../../include/core/Font.h"

#include "../../include/core/Entity.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Rigidbody.h"
#include "../../include/components/Camera.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Light.h"
#include "../../include/components/LineRenderer.h"
#include "../../include/components/BoxCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/CapsuleCollider.h"
#include "../../include/components/MeshCollider.h"



using namespace PhysicsEngine;
using namespace json;


void PhysicsEngine::writeInternalAssetToJson(json::JSON& obj, World* world, Guid assetId, int type)
{
	// only materials are stored in json format
	if (type == 4) {
		//material
		Material* material = world->getAsset<Material>(assetId);

		obj["shader"] = material->getShaderId().toString();

		std::vector<ShaderUniform> uniforms = material->getUniforms();
		for (size_t i = 0; i < uniforms.size(); i++) {
			if (uniforms[i].type == GL_INT) {
				obj[uniforms[i].name]["data"] = *reinterpret_cast<int*>(uniforms[i].data);
			}
			else if (uniforms[i].type == GL_FLOAT) {
				obj[uniforms[i].name]["data"] = *reinterpret_cast<float*>(uniforms[i].data);
			}
			else if (uniforms[i].type == GL_FLOAT_VEC2) {
				glm::vec2 data = *reinterpret_cast<glm::vec2*>(uniforms[i].data);
				obj[uniforms[i].name]["data"].append(data.x, data.y);
			}
			else if (uniforms[i].type == GL_FLOAT_VEC3) {
				glm::vec3 data = *reinterpret_cast<glm::vec3*>(uniforms[i].data);
				obj[uniforms[i].name]["data"].append(data.x, data.y, data.z);
			}
			else if (uniforms[i].type == GL_FLOAT_VEC4) {
				glm::vec4 data = *reinterpret_cast<glm::vec4*>(uniforms[i].data);
				obj[uniforms[i].name]["data"].append(data.x, data.y, data.z, data.w);
			}

			if (uniforms[i].type == GL_SAMPLER_2D) {
				Guid textureId = *reinterpret_cast<Guid*>(uniforms[i].data);
				obj[uniforms[i].name]["data"] = textureId.toString();
			}

			obj[uniforms[i].name]["shortName"] = uniforms[i].shortName;
			obj[uniforms[i].name]["blockName"] = uniforms[i].blockName;
			obj[uniforms[i].name]["nameLength"] = (int)uniforms[i].nameLength;
			obj[uniforms[i].name]["size"] = (int)uniforms[i].size;
			obj[uniforms[i].name]["type"] = (int)uniforms[i].type;
			obj[uniforms[i].name]["variant"] = uniforms[i].variant;
			obj[uniforms[i].name]["location"] = uniforms[i].location;
			obj[uniforms[i].name]["index"] = (int)uniforms[i].index;
		}
	}
}

void PhysicsEngine::writeInternalEntityToJson(json::JSON& obj, World* world, Guid entityId)
{
	Entity* entity = world->getEntity(entityId);

	std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);

	// write entity to json
	obj[entityId.toString()] = json::Object();
	obj[entityId.toString()]["type"] = "Entity";
	for (size_t j = 0; j < componentsOnEntity.size(); j++) {
		obj[entityId.toString()]["components"].append(componentsOnEntity[j].first.toString());
	}
}

void PhysicsEngine::writeInternalComponentToJson(json::JSON& obj, World* world, Guid entityId, Guid componentId, int type)
{
	if (type > 20) {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to write internal component to json\n";
		Log::error(message.c_str());
		return;
	}

	if (type == 0) {
		//transform
		Transform* transform = world->getComponent<Transform>(entityId);

		obj[componentId.toString()]["type"] = "Transform";
		obj[componentId.toString()]["parent"] = transform->parentId.toString();
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["position"].append(transform->position.x, transform->position.y, transform->position.z);
		obj[componentId.toString()]["rotation"].append(transform->rotation.x, transform->rotation.y, transform->rotation.z, transform->rotation.w);
		obj[componentId.toString()]["scale"].append(transform->scale.x, transform->scale.y, transform->scale.z);
	}
	else if (type == 1) {
		//rigidbody
		Rigidbody* rigidbody = world->getComponent<Rigidbody>(entityId);

		obj[componentId.toString()]["type"] = "Rigidbody";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["useGravity"] = rigidbody->useGravity;
		obj[componentId.toString()]["mass"] = rigidbody->mass;
		obj[componentId.toString()]["drag"] = rigidbody->drag;
		obj[componentId.toString()]["angularDrag"] = rigidbody->angularDrag;
	}
	else if (type == 2) {
		//camera
		Camera* camera = world->getComponent<Camera>(entityId);

		obj[componentId.toString()]["type"] = "Camera";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["targetTextureId"] = camera->targetTextureId.toString();
		obj[componentId.toString()]["position"].append(camera->position.x, camera->position.y, camera->position.z);
		obj[componentId.toString()]["front"].append(camera->front.x, camera->front.y, camera->front.z);
		obj[componentId.toString()]["up"].append(camera->up.x, camera->up.y, camera->up.z);
		obj[componentId.toString()]["backgroundColor"].append(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
		obj[componentId.toString()]["x"] = camera->viewport.x;
		obj[componentId.toString()]["y"] = camera->viewport.y;
		obj[componentId.toString()]["width"] = camera->viewport.width;
		obj[componentId.toString()]["height"] = camera->viewport.height;
		obj[componentId.toString()]["fov"] = camera->frustum.fov;
		obj[componentId.toString()]["near"] = camera->frustum.nearPlane;
		obj[componentId.toString()]["far"] = camera->frustum.farPlane;
	}
	else if (type == 3) {
		//meshrenderer
		MeshRenderer* meshRenderer = world->getComponent<MeshRenderer>(entityId);

		obj[componentId.toString()]["type"] = "MeshRenderer";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["mesh"] = meshRenderer->meshId.toString();

		int materialCount = meshRenderer->materialCount;

		std::string label = "material";
		if (materialCount > 1) {
			label = "materials";
		}

		std::string value = "";
		if (materialCount == 0) {
			value = Guid::INVALID.toString();
		}
		else if (materialCount == 1) {
			value = meshRenderer->materialIds[0].toString();
		}
		else { // dont think this is right. I think I need to do something like obj[componentId.toString()][label].append...
			value += "[";
			for (int m = 0; m < materialCount; m++) {
				value += meshRenderer->materialIds[m].toString();
				if (m != materialCount - 1) {
					value += ",";
				}
			}
			value += "]";
		}

		obj[componentId.toString()][label] = value;
		obj[componentId.toString()]["isStatic"] = meshRenderer->isStatic;
	}
	else if (type == 4) {
		//linerenderer
		LineRenderer* lineRenderer = world->getComponent<LineRenderer>(entityId);
	}
	else if (type == 5) {
		//light
		Light* light = world->getComponent<Light>(entityId);

		obj[componentId.toString()]["type"] = "Light";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["position"].append(light->position.x, light->position.y, light->position.z);
		obj[componentId.toString()]["direction"].append(light->direction.x, light->direction.y, light->direction.z);
		obj[componentId.toString()]["ambient"].append(light->ambient.x, light->ambient.y, light->ambient.z);
		obj[componentId.toString()]["diffuse"].append(light->diffuse.x, light->diffuse.y, light->diffuse.z);
		obj[componentId.toString()]["specular"].append(light->specular.x, light->specular.y, light->specular.z);
		obj[componentId.toString()]["constant"] = light->constant;
		obj[componentId.toString()]["linear"] = light->linear;
		obj[componentId.toString()]["quadratic"] = light->quadratic;
		obj[componentId.toString()]["cutOff"] = light->cutOff;
		obj[componentId.toString()]["outerCutOff"] = light->outerCutOff;
		obj[componentId.toString()]["lightType"] = static_cast<int>(light->lightType);
		obj[componentId.toString()]["shadowType"] = static_cast<int>(light->shadowType);
	}
	else if (type == 8) {
		//boxcollider
		BoxCollider* collider = world->getComponent<BoxCollider>(entityId);

		obj[componentId.toString()]["type"] = "SphereCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->bounds.centre.x, collider->bounds.centre.y, collider->bounds.centre.z);
		obj[componentId.toString()]["size"].append(collider->bounds.size.x, collider->bounds.size.y, collider->bounds.size.z);
	}
	else if (type == 9) {
		//spherecollider
		SphereCollider* collider = world->getComponent<SphereCollider>(entityId);

		obj[componentId.toString()]["type"] = "SphereCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->sphere.centre.x, collider->sphere.centre.y, collider->sphere.centre.z);
		obj[componentId.toString()]["radius"] = collider->sphere.radius;
	}
	else if (type == 15) {
		//meshcollider
		MeshCollider* collider = world->getComponent<MeshCollider>(entityId);
	}
	else if (type == 10) {
		//capsulecolldier
		CapsuleCollider* collider = world->getComponent<CapsuleCollider>(entityId);

		obj[componentId.toString()]["type"] = "CapsuleCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->capsule.centre.x, collider->capsule.centre.y, collider->capsule.centre.z);
		obj[componentId.toString()]["radius"] = collider->capsule.radius;
		obj[componentId.toString()]["height"] = collider->capsule.height;
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to write internal component to json\n";
		Log::error(message.c_str());
		return;
	}
}



void PhysicsEngine::writeInternalSystemToJson(json::JSON& obj, World* world, Guid systemId, int type, int order)
{
	if (type > 20) {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to write internal system to json\n";
		Log::error(message.c_str());
		return;
	}

	if (type == 0) {
		// RenderSystem
		obj[systemId.toString()]["type"] = "RenderSystem";
		obj[systemId.toString()]["order"] = order;

	}
	else if (type == 1) {
		// PhysicsSystem
		obj[systemId.toString()]["type"] = "PhysicsSystem";
		obj[systemId.toString()]["order"] = order;
	}
	else if (type == 2) {
		// CleanUpSystem
		obj[systemId.toString()]["type"] = "CleanUpSystem";
		obj[systemId.toString()]["order"] = order;
	}
	else if (type == 3) {
		// DebugSystem
		obj[systemId.toString()]["type"] = "DebugSystem";
		obj[systemId.toString()]["order"] = order;
	}
	else {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
		Log::error(message.c_str());
		return;
	}
}