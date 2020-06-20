#include <iostream>
#include <cstddef>
#include <ctime>
#include <random>
#include <unordered_set>

#include "../../include/core/Shader.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/core/Geometry.h"

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/DeferredRenderer.h"

#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	mRenderToScreen = true;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	deserialize(data);
}

RenderSystem::~RenderSystem()
{
}

std::vector<char> RenderSystem::serialize() const
{
	return serialize(mSystemId);
}

std::vector<char> RenderSystem::serialize(Guid systemId) const
{
	std::vector<char> data(sizeof(int));

	memcpy(&data[0], &mOrder, sizeof(int));

	return data;
}

void RenderSystem::deserialize(std::vector<char> data)
{
	mOrder = *reinterpret_cast<int*>(&data[0]);
}

void RenderSystem::init(World* world)
{
	mWorld = world;

	mForwardRenderer.init(mWorld, mRenderToScreen);
	mDeferredRenderer.init(mWorld, mRenderToScreen);
}

void RenderSystem::update(Input input, Time time)
{
	registerRenderAssets(mWorld);
	registerCameras(mWorld);

	updateRenderObjects(mWorld, mRenderObjects);
	updateModelMatrices(mWorld, mRenderObjects);

	const PoolAllocator<Camera>* cameraAllocator = mWorld->getComponentAllocator_Const<Camera>();

	for (int i = 0; i < mWorld->getNumberOfComponents<Camera>(cameraAllocator); i++) {
		Camera* camera = mWorld->getComponentByIndex<Camera>(cameraAllocator, i);

		cullRenderObjects(camera, mRenderObjects);

		if (camera->mRenderPath == RenderPath::Forward) {
			mForwardRenderer.update(input, camera, mRenderObjects);
		}
		else {
			mDeferredRenderer.update(input, camera, mRenderObjects);
		}
	}
}

void RenderSystem::registerRenderAssets(World* world)
{
	// create all texture assets not already created
	for (int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++) {
		Texture2D* texture = world->getAssetByIndex<Texture2D>(i);
		if (texture != NULL && !texture->isCreated()) {
			texture->create();

			if (!texture->isCreated()) {
				std::string errorMessage = "Error: Failed to create texture " + texture->getId().toString() + "\n";
				Log::error(errorMessage.c_str());
			}
		}
	}

	// compile all shader assets and configure uniform blocks not already compiled
	std::unordered_set<Guid> shadersCompiledThisFrame;
	for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
		Shader* shader = world->getAssetByIndex<Shader>(i);

		if (!shader->isCompiled()) {

			shader->compile();

			if (!shader->isCompiled()) {
				std::string errorMessage = "Shader failed to compile " + shader->getId().toString() + "\n";
				Log::error(&errorMessage[0]);
			}

			shadersCompiledThisFrame.insert(shader->getId());
		}
	}

	// update material on shader change
	for (int i = 0; i < world->getNumberOfAssets<Material>(); i++) {
		Material* material = world->getAssetByIndex<Material>(i);

		std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderId());

		if (material->hasShaderChanged() || it != shadersCompiledThisFrame.end()) {
			material->onShaderChanged(world); // need to also do this if the shader code changed but the assigned shader on the material remained the same!
		}
	}

	// create all mesh assets not already created
	for (int i = 0; i < world->getNumberOfAssets<Mesh>(); i++) {
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		if (mesh != NULL && !mesh->isCreated()) {
			mesh->create();

			if (!mesh->isCreated()) {
				std::string errorMessage = "Error: Failed to create mesh " + mesh->getId().toString() + "\n";
				Log::error(errorMessage.c_str());
			}
		}
	}
}

void RenderSystem::registerCameras(World* world)
{
	const PoolAllocator<Camera>* cameraAllocator = world->getComponentAllocator_Const<Camera>();

	for (int i = 0; i < world->getNumberOfComponents<Camera>(); i++) {
		Camera* camera = world->getComponentByIndex<Camera>(cameraAllocator, i);

		if (!camera->isCreated()) {
			camera->create();
		}
	}
}

void RenderSystem::updateRenderObjects(World* world, std::vector<RenderObject>& renderObjects)
{
	const PoolAllocator<MeshRenderer>* meshRendererAllocator = world->getComponentAllocator_Const<MeshRenderer>();

	// add created mesh renderers to render object list
	std::vector<triple<Guid, Guid, int>> componentIdsAdded = world->getComponentIdsMarkedCreated();
	for (size_t i = 0; i < componentIdsAdded.size(); i++) {

		if (componentIdsAdded[i].third == ComponentType<MeshRenderer>::type) {
			int meshRendererIndex = world->getIndexOf(componentIdsAdded[i].second);

			MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(meshRendererAllocator, meshRendererIndex);
			Transform* transform = meshRenderer->getComponent<Transform>(world);
			Mesh* mesh = world->getAsset<Mesh>(meshRenderer->getMesh());

			int transformIndex = world->getIndexOf(transform->getId());
			int meshIndex = mesh != NULL ? world->getIndexOf(mesh->getId()) : -1;

			Sphere boundingSphere = mesh != NULL ? mesh->computeBoundingSphere() : Sphere();

			for (int j = 0; j < 8; j++) {
				int materialIndex = world->getIndexOf(meshRenderer->getMaterial(j));
				Material* material = world->getAssetByIndex<Material>(materialIndex);

				int shaderIndex = material != NULL ? world->getIndexOf(material->getShaderId()) : -1;
				Shader* shader = material != NULL ? world->getAssetByIndex<Shader>(shaderIndex) : NULL;

				RenderObject renderObject;
				renderObject.meshRendererId = meshRenderer->getId();
				renderObject.transformId = transform->getId();
				renderObject.meshId = mesh != NULL ? mesh->getId() : Guid::INVALID;
				renderObject.materialId = material != NULL ? material->getId() : Guid::INVALID;
				renderObject.shaderId = shader != NULL ? shader->getId() : Guid::INVALID;

				renderObject.meshRendererIndex = meshRendererIndex;
				renderObject.transformIndex = transformIndex;
				renderObject.meshIndex = meshIndex;
				renderObject.materialIndex = materialIndex;
				renderObject.shaderIndex = shaderIndex;
				renderObject.subMeshIndex = j;

				renderObject.boundingSphere = boundingSphere;

				int subMeshVertexStartIndex = mesh != NULL ? mesh->getSubMeshStartIndex(j) : -1;
				int subMeshVertexEndIndex = mesh != NULL ? mesh->getSubMeshEndIndex(j) : -1;

				renderObject.start = subMeshVertexStartIndex;
				renderObject.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
				renderObject.vao = mesh != NULL ? mesh->getNativeGraphicsVAO() : -1;

				renderObjects.push_back(renderObject);
			}

			meshRenderer->mMeshChanged = false;
			meshRenderer->mMaterialChanged = false;
		}
	}

	// remove destroyed mesh renderers from render object list
	std::vector<triple<Guid, Guid, int>> componentIdsDestroyed = world->getComponentIdsMarkedLatentDestroy();
	for (size_t i = 0; i < componentIdsDestroyed.size(); i++) {
		if (componentIdsDestroyed[i].third == ComponentType<MeshRenderer>::type) {
			int meshRendererIndex = world->getIndexOf(componentIdsDestroyed[i].second);

			//mmm this is slow...need a faster way of removing render objects
			for (int j = (int)renderObjects.size() - 1; j >= 0; j--) {
				if (meshRendererIndex == renderObjects[j].meshRendererIndex) {
					renderObjects.erase(renderObjects.begin() + j);
				}
			}
		}
	}

	// update render object list for transforms and/or mesh renderers that have been moved in their global arrays
	std::vector<triple<Guid, int, int>> componentIdsMoved = world->getComponentIdsMarkedMoved();
	for (size_t i = 0; i < componentIdsMoved.size(); i++) {
		if (componentIdsMoved[i].second == ComponentType<Transform>::type) {
			int oldIndex = componentIdsMoved[i].third;
			int newIndex = world->getIndexOf(componentIdsMoved[i].first);

			//mmm this is slow...
			for (size_t j = 0; j < renderObjects.size(); j++) {
				if (oldIndex == renderObjects[j].transformIndex) {
					renderObjects[j].transformIndex = newIndex;
				}
			}
		}
		else if (componentIdsMoved[i].second == ComponentType<MeshRenderer>::type)
		{
			int oldIndex = componentIdsMoved[i].third;
			int newIndex = world->getIndexOf(componentIdsMoved[i].first);

			//mmm this is slow...
			for (size_t j = 0; j < renderObjects.size(); j++) {
				if (oldIndex == renderObjects[j].meshRendererIndex) {
					renderObjects[j].meshRendererIndex = newIndex;
				}
			}
		}
	}

	// update render objects list for changes in mesh and material
	for (size_t i = 0; i < renderObjects.size(); i++) {

		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(meshRendererAllocator, renderObjects[i].meshRendererIndex);

		// update render objects list for mesh renderers whose mesh has changed
		if (meshRenderer->mMeshChanged) {

			int meshIndex = world->getIndexOf(meshRenderer->getMesh());

			Mesh* mesh = world->getAssetByIndex<Mesh>(meshIndex);

			renderObjects[i].meshId = meshRenderer->getMesh();
			renderObjects[i].meshIndex = meshIndex;

			int subMeshIndex = renderObjects[i].subMeshIndex;
			int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(subMeshIndex);
			int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(subMeshIndex);

			renderObjects[i].start = subMeshVertexStartIndex;
			renderObjects[i].size = subMeshVertexEndIndex - subMeshVertexStartIndex;
			renderObjects[i].vao = mesh != NULL ? mesh->getNativeGraphicsVAO() : -1;

			renderObjects[i].boundingSphere = mesh->computeBoundingSphere();

			meshRenderer->mMeshChanged = false;
		}

		// update render objects list for mesh renderers whose material has changed
		if (meshRenderer->mMaterialChanged) {

			int materialIndex = world->getIndexOf(meshRenderer->getMaterial(renderObjects[i].subMeshIndex));
			Material* material = world->getAssetByIndex<Material>(materialIndex);

			int shaderIndex = material != NULL ? world->getIndexOf(material->getShaderId()) : -1;

			renderObjects[i].materialId = material != NULL ? material->getId() : Guid::INVALID;
			renderObjects[i].shaderId = material != NULL ? material->getShaderId() : Guid::INVALID;
			renderObjects[i].materialIndex = materialIndex;
			renderObjects[i].shaderIndex = shaderIndex;

			meshRenderer->mMaterialChanged = false;
		}
	}
}

void RenderSystem::cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects)
{
	int count = 0;
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		glm::vec3 centre = renderObjects[i].boundingSphere.mCentre;
		float radius = renderObjects[i].boundingSphere.mRadius;

		glm::vec4 temp = renderObjects[i].model * glm::vec4(centre.x, centre.y, centre.z, 1.0f);

		Sphere cullingSphere;
		cullingSphere.mCentre = glm::vec3(temp.x, temp.y, temp.z);
		cullingSphere.mRadius = radius;

		if (Geometry::intersect(cullingSphere, camera->mFrustum)) {
			count++;
		}
	}
}

void RenderSystem::updateModelMatrices(World* world, std::vector<RenderObject>& renderObjects)
{
	const PoolAllocator<Transform>* transformAllocator = world->getComponentAllocator_Const<Transform>();

	// update model matrices
	int n = (int)renderObjects.size();

	for (int i = 0; i < n; i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		Transform* transform = world->getComponentByIndex<Transform>(transformAllocator, renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();
	}
}