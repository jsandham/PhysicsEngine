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
			if (uniforms[i].mType == GL_INT) {
				obj[uniforms[i].mName]["data"] = *reinterpret_cast<int*>(uniforms[i].mData);
			}
			else if (uniforms[i].mType == GL_FLOAT) {
				obj[uniforms[i].mName]["data"] = *reinterpret_cast<float*>(uniforms[i].mData);
			}
			else if (uniforms[i].mType == GL_FLOAT_VEC2) {
				glm::vec2 data = *reinterpret_cast<glm::vec2*>(uniforms[i].mData);
				obj[uniforms[i].mName]["data"].append(data.x, data.y);
			}
			else if (uniforms[i].mType == GL_FLOAT_VEC3) {
				glm::vec3 data = *reinterpret_cast<glm::vec3*>(uniforms[i].mData);
				obj[uniforms[i].mName]["data"].append(data.x, data.y, data.z);
			}
			else if (uniforms[i].mType == GL_FLOAT_VEC4) {
				glm::vec4 data = *reinterpret_cast<glm::vec4*>(uniforms[i].mData);
				obj[uniforms[i].mName]["data"].append(data.x, data.y, data.z, data.w);
			}

			if (uniforms[i].mType == GL_SAMPLER_2D) {
				Guid textureId = *reinterpret_cast<Guid*>(uniforms[i].mData);
				obj[uniforms[i].mName]["data"] = textureId.toString();
			}

			obj[uniforms[i].mName]["shortName"] = uniforms[i].mShortName;
			obj[uniforms[i].mName]["blockName"] = uniforms[i].mBlockName;
			obj[uniforms[i].mName]["nameLength"] = (int)uniforms[i].mNameLength;
			obj[uniforms[i].mName]["size"] = (int)uniforms[i].mSize;
			obj[uniforms[i].mName]["type"] = (int)uniforms[i].mType;
			obj[uniforms[i].mName]["variant"] = uniforms[i].mVariant;
			obj[uniforms[i].mName]["location"] = uniforms[i].mLocation;
			obj[uniforms[i].mName]["index"] = (int)uniforms[i].mIndex;
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
		obj[componentId.toString()]["parent"] = transform->mParentId.toString();
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["position"].append(transform->mPosition.x, transform->mPosition.y, transform->mPosition.z);
		obj[componentId.toString()]["rotation"].append(transform->mRotation.x, transform->mRotation.y, transform->mRotation.z, transform->mRotation.w);
		obj[componentId.toString()]["scale"].append(transform->mScale.x, transform->mScale.y, transform->mScale.z);
	}
	else if (type == 1) {
		//rigidbody
		Rigidbody* rigidbody = world->getComponent<Rigidbody>(entityId);

		obj[componentId.toString()]["type"] = "Rigidbody";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["useGravity"] = rigidbody->mUseGravity;
		obj[componentId.toString()]["mass"] = rigidbody->mMass;
		obj[componentId.toString()]["drag"] = rigidbody->mDrag;
		obj[componentId.toString()]["angularDrag"] = rigidbody->mAngularDrag;
	}
	else if (type == 2) {
		//camera
		Camera* camera = world->getComponent<Camera>(entityId);

		obj[componentId.toString()]["type"] = "Camera";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["targetTextureId"] = camera->mTargetTextureId.toString();
		obj[componentId.toString()]["position"].append(camera->mPosition.x, camera->mPosition.y, camera->mPosition.z);
		obj[componentId.toString()]["front"].append(camera->mFront.x, camera->mFront.y, camera->mFront.z);
		obj[componentId.toString()]["up"].append(camera->mUp.x, camera->mUp.y, camera->mUp.z);
		obj[componentId.toString()]["backgroundColor"].append(camera->mBackgroundColor.x, camera->mBackgroundColor.y, camera->mBackgroundColor.z, camera->mBackgroundColor.w);
		obj[componentId.toString()]["x"] = camera->mViewport.mX;
		obj[componentId.toString()]["y"] = camera->mViewport.mY;
		obj[componentId.toString()]["width"] = camera->mViewport.mWidth;
		obj[componentId.toString()]["height"] = camera->mViewport.mHeight;
		obj[componentId.toString()]["fov"] = camera->mFrustum.mFov;
		obj[componentId.toString()]["near"] = camera->mFrustum.mNearPlane;
		obj[componentId.toString()]["far"] = camera->mFrustum.mFarPlane;
	}
	else if (type == 3) {
		//meshrenderer
		MeshRenderer* meshRenderer = world->getComponent<MeshRenderer>(entityId);

		obj[componentId.toString()]["type"] = "MeshRenderer";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["mesh"] = meshRenderer->mMeshId.toString();

		int materialCount = meshRenderer->mMaterialCount;

		std::string label = "material";
		if (materialCount > 1) {
			label = "materials";
		}

		std::string value = "";
		if (materialCount == 0) {
			value = Guid::INVALID.toString();
		}
		else if (materialCount == 1) {
			value = meshRenderer->mMaterialIds[0].toString();
		}
		else { // dont think this is right. I think I need to do something like obj[componentId.toString()][label].append...
			value += "[";
			for (int m = 0; m < materialCount; m++) {
				value += meshRenderer->mMaterialIds[m].toString();
				if (m != materialCount - 1) {
					value += ",";
				}
			}
			value += "]";
		}

		obj[componentId.toString()][label] = value;
		obj[componentId.toString()]["isStatic"] = meshRenderer->mIsStatic;
	}
	else if (type == 4) {
		//linerenderer
		//LineRenderer* lineRenderer = world->getComponent<LineRenderer>(entityId);
	}
	else if (type == 5) {
		//light
		Light* light = world->getComponent<Light>(entityId);

		obj[componentId.toString()]["type"] = "Light";
		obj[componentId.toString()]["entity"] = entityId.toString();
		obj[componentId.toString()]["position"].append(light->mPosition.x, light->mPosition.y, light->mPosition.z);
		obj[componentId.toString()]["direction"].append(light->mDirection.x, light->mDirection.y, light->mDirection.z);
		obj[componentId.toString()]["ambient"].append(light->mAmbient.x, light->mAmbient.y, light->mAmbient.z);
		obj[componentId.toString()]["diffuse"].append(light->mDiffuse.x, light->mDiffuse.y, light->mDiffuse.z);
		obj[componentId.toString()]["specular"].append(light->mSpecular.x, light->mSpecular.y, light->mSpecular.z);
		obj[componentId.toString()]["constant"] = light->mConstant;
		obj[componentId.toString()]["linear"] = light->mLinear;
		obj[componentId.toString()]["quadratic"] = light->mQuadratic;
		obj[componentId.toString()]["cutOff"] = light->mCutOff;
		obj[componentId.toString()]["outerCutOff"] = light->mOuterCutOff;
		obj[componentId.toString()]["lightType"] = static_cast<int>(light->mLightType);
		obj[componentId.toString()]["shadowType"] = static_cast<int>(light->mShadowType);
	}
	else if (type == 8) {
		//boxcollider
		BoxCollider* collider = world->getComponent<BoxCollider>(entityId);

		obj[componentId.toString()]["type"] = "SphereCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->mBounds.mCentre.x, collider->mBounds.mCentre.y, collider->mBounds.mCentre.z);
		obj[componentId.toString()]["size"].append(collider->mBounds.mSize.x, collider->mBounds.mSize.y, collider->mBounds.mSize.z);
	}
	else if (type == 9) {
		//spherecollider
		SphereCollider* collider = world->getComponent<SphereCollider>(entityId);

		obj[componentId.toString()]["type"] = "SphereCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->mSphere.mCentre.x, collider->mSphere.mCentre.y, collider->mSphere.mCentre.z);
		obj[componentId.toString()]["radius"] = collider->mSphere.mRadius;
	}
	else if (type == 15) {
		//meshcollider
		//MeshCollider* collider = world->getComponent<MeshCollider>(entityId);
	}
	else if (type == 10) {
		//capsulecolldier
		CapsuleCollider* collider = world->getComponent<CapsuleCollider>(entityId);

		obj[componentId.toString()]["type"] = "CapsuleCollider";
		obj[componentId.toString()]["entity"] = entityId.toString();

		obj[componentId.toString()]["centre"].append(collider->mCapsule.mCentre.x, collider->mCapsule.mCentre.y, collider->mCapsule.mCentre.z);
		obj[componentId.toString()]["radius"] = collider->mCapsule.mRadius;
		obj[componentId.toString()]["height"] = collider->mCapsule.mHeight;
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