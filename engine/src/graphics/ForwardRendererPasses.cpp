#include <random>
#include <unordered_set>

#include "../../include/core/Shader.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/core/Geometry.h"

#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

void PhysicsEngine::initializeRenderer(World* world, ForwardRendererState* state)
{
	state->mQuery.mQueryBack = 0;
	state->mQuery.mQueryFront = 1;

	glGenQueries(1, &(state->mQuery.mQueryId[0]));
	glGenQueries(1, &(state->mQuery.mQueryId[1]));

	// generate all internal shader programs
	state->mGeometryShader.setVertexShader(InternalShaders::positionAndNormalsVertexShader);
	state->mGeometryShader.setFragmentShader(InternalShaders::positionAndNormalsFragmentShader);
	state->mGeometryShader.compile();

	state->mColorShader.setVertexShader(InternalShaders::colorVertexShader);
	state->mColorShader.setFragmentShader(InternalShaders::colorFragmentShader);
	state->mColorShader.compile();

	state->mSsaoShader.setVertexShader(InternalShaders::ssaoVertexShader);
	state->mSsaoShader.setFragmentShader(InternalShaders::ssaoFragmentShader);
	state->mSsaoShader.compile();

	state->mDepthShader.setVertexShader(InternalShaders::shadowDepthMapVertexShader);
	state->mDepthShader.setFragmentShader(InternalShaders::shadowDepthMapFragmentShader);
	state->mDepthShader.compile();

	state->mDepthCubemapShader.setVertexShader(InternalShaders::shadowDepthCubemapVertexShader);
	state->mDepthCubemapShader.setFragmentShader(InternalShaders::shadowDepthCubemapFragmentShader);
	state->mDepthCubemapShader.compile();

	state->mQuadShader.setVertexShader(InternalShaders::windowVertexShader);
	state->mQuadShader.setFragmentShader(InternalShaders::windowFragmentShader);
	state->mQuadShader.compile();

	// cache internal shader uniforms
	state->mGeometryShaderProgram = state->mGeometryShader.getProgramFromVariant(ShaderVariant::None);
	state->mGeometryShaderModelLoc = state->mGeometryShader.findUniformLocation("model", state->mGeometryShaderProgram);

	state->mSsaoShaderProgram = state->mSsaoShader.getProgramFromVariant(ShaderVariant::None);
	state->mSsaoShaderProjectionLoc = state->mSsaoShader.findUniformLocation("projection", state->mSsaoShaderProgram);
	state->mSsaoShaderPositionTexLoc = state->mSsaoShader.findUniformLocation("positionTex", state->mSsaoShaderProgram);
	state->mSsaoShaderNormalTexLoc = state->mSsaoShader.findUniformLocation("normalTex", state->mSsaoShaderProgram);
	state->mSsaoShaderNoiseTexLoc = state->mSsaoShader.findUniformLocation("noiseTex", state->mSsaoShaderProgram);

	for (int i = 0; i < 64; i++) {
		state->mSsaoShaderSamplesLoc[i] = state->mSsaoShader.findUniformLocation("samples[" + std::to_string(i) + "]", state->mSsaoShaderProgram);
	}

	state->mDepthShaderProgram = state->mDepthShader.getProgramFromVariant(ShaderVariant::None);
	state->mDepthShaderModelLoc = state->mDepthShader.findUniformLocation("model", state->mDepthShaderProgram);
	state->mDepthShaderViewLoc = state->mDepthShader.findUniformLocation("view", state->mDepthShaderProgram);
	state->mDepthShaderProjectionLoc = state->mDepthShader.findUniformLocation("projection", state->mDepthShaderProgram);

	state->mDepthCubemapShaderProgram = state->mDepthCubemapShader.getProgramFromVariant(ShaderVariant::None);
	state->mDepthCubemapShaderLightPosLoc = state->mDepthCubemapShader.findUniformLocation("lightPos", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderFarPlaneLoc = state->mDepthCubemapShader.findUniformLocation("farPlane", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderModelLoc = state->mDepthCubemapShader.findUniformLocation("model", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc0 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[0]", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc1 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[1]", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc2 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[2]", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc3 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[3]", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc4 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[4]", state->mDepthCubemapShaderProgram);
	state->mDepthCubemapShaderCubeViewProjMatricesLoc5 = state->mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[5]", state->mDepthCubemapShaderProgram);

	state->mColorShaderProgram = state->mColorShader.getProgramFromVariant(ShaderVariant::None);
	state->mColorShaderModelLoc = state->mColorShader.findUniformLocation("model", state->mColorShaderProgram);
	state->mColorShaderColorLoc = state->mColorShader.findUniformLocation("color", state->mColorShaderProgram);

	Graphics::checkError();

	//generate screen quad for final rendering
	constexpr float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};

	glGenVertexArrays(1, &state->mQuadVAO);
	glBindVertexArray(state->mQuadVAO);

	glGenBuffers(1, &state->mQuadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, state->mQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	Graphics::checkError();

	// generate shadow map fbos
	// create directional light cascade shadow map fbo
	glGenFramebuffers(5, &state->mShadowCascadeFBO[0]);
	glGenTextures(5, &state->mShadowCascadeDepth[0]);

	for (int i = 0; i < 5; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowCascadeFBO[i]);
		glBindTexture(GL_TEXTURE_2D, state->mShadowCascadeDepth[i]);
		// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, state->mShadowCascadeDepth[i], 0);

		Graphics::checkFrambufferError();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// create spotlight shadow map fbo
	glGenFramebuffers(1, &state->mShadowSpotlightFBO);
	glGenTextures(1, &state->mShadowSpotlightDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowSpotlightFBO);
	glBindTexture(GL_TEXTURE_2D, state->mShadowSpotlightDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, state->mShadowSpotlightDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create pointlight shadow cubemap fbo
	glGenFramebuffers(1, &state->mShadowCubemapFBO);
	glGenTextures(1, &state->mShadowCubemapDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowCubemapFBO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, state->mShadowCubemapDepth);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, state->mShadowCubemapDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	glGenBuffers(1, &(state->mCameraState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(state->mLightState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
	glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::registerRenderAssets(World* world)
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

	Graphics::checkError();

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

	Graphics::checkError();

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

	Graphics::checkError();
}

void PhysicsEngine::registerCameras(World* world)
{
	const PoolAllocator<Camera>* cameraAllocator = world->getComponentAllocator_Const<Camera>();

	for (int i = 0; i < world->getNumberOfComponents<Camera>(); i++) {
		Camera* camera = world->getComponentByIndex<Camera>(cameraAllocator, i);

		if (!camera->isCreated()) {
			camera->create();
		}
	}

	Graphics::checkError();
}

void PhysicsEngine::updateRenderObjects(World* world, std::vector<RenderObject>& renderObjects)
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

void PhysicsEngine::cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects)
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

void PhysicsEngine::updateModelMatrices(World* world, std::vector<RenderObject>& renderObjects)
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

void PhysicsEngine::beginFrame(World* world, Camera* camera, ForwardRendererState* state)
{
	state->mQuery.mNumBatchDrawCalls = 0;
	state->mQuery.mNumDrawCalls = 0;
	state->mQuery.mTotalElapsedTime = 0.0f;
	state->mQuery.mVerts = 0;
	state->mQuery.mTris = 0;
	state->mQuery.mLines = 0;
	state->mQuery.mPoints = 0;

	glBeginQuery(GL_TIME_ELAPSED, state->mQuery.mQueryId[state->mQuery.mQueryBack]);

	state->mCameraState.mProjection = camera->getProjMatrix();
	state->mCameraState.mView = camera->getViewMatrix();
	state->mCameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

	// set camera state binding point and update camera state data
	glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->mCameraState.mHandle, 0, 144);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state->mCameraState.mProjection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state->mCameraState.mView));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(state->mCameraState.mCameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// set light state binding point
	glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, state->mLightState.mHandle, 0, 824);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glEnable(GL_SCISSOR_TEST);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);

	glViewport(camera->mViewport.mX, camera->mViewport.mY, camera->mViewport.mWidth, camera->mViewport.mHeight);
	glScissor(camera->mViewport.mX, camera->mViewport.mY, camera->mViewport.mWidth, camera->mViewport.mHeight);

	glClearColor(camera->mBackgroundColor.x, camera->mBackgroundColor.y, camera->mBackgroundColor.z, camera->mBackgroundColor.w);
	glClearDepth(1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// This is currently really expensive so I have it commented out right now
void PhysicsEngine::computeSSAO(World* world,
	Camera* camera,
	ForwardRendererState* state,
	const std::vector<RenderObject>& renderObjects)
{
	// fill geometry framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		state->mGeometryShader.use(state->mGeometryShaderProgram);
		state->mGeometryShader.setMat4(state->mGeometryShaderModelLoc, renderObjects[i].model);

		Graphics::render(world, renderObjects[i], &state->mQuery);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	// fill ssao color texture
	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());
	state->mSsaoShader.use(state->mSsaoShaderProgram);
	state->mSsaoShader.setMat4(state->mSsaoShaderProjectionLoc, camera->getProjMatrix());
	for (int i = 0; i < 64; i++) {
		state->mSsaoShader.setVec3(state->mSsaoShaderSamplesLoc[i], camera->getSSAOSample(i));
	}
	state->mSsaoShader.setInt(state->mSsaoShaderPositionTexLoc, 0);
	state->mSsaoShader.setInt(state->mSsaoShaderNormalTexLoc, 1);
	state->mSsaoShader.setInt(state->mSsaoShaderNoiseTexLoc, 2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsPositionTex());
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsNormalTex());
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsSSAONoiseTex());

	glBindVertexArray(state->mQuadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderShadows(World* world,
	Camera* camera,
	Light* light,
	Transform* lightTransform,
	ForwardRendererState* state,
	const std::vector<RenderObject>& renderObjects)
{
	if (light->mLightType == LightType::Directional) {

		calcShadowmapCascades(camera, state);
		calcCascadeOrthoProj(camera, lightTransform->getForward(), state);

		for (int i = 0; i < 5; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowCascadeFBO[i]);

			glClearDepth(1.0f);
			glClear(GL_DEPTH_BUFFER_BIT);

			state->mDepthShader.use(state->mDepthShaderProgram);
			state->mDepthShader.setMat4(state->mDepthShaderViewLoc, state->mCascadeLightView[i]);
			state->mDepthShader.setMat4(state->mDepthShaderProjectionLoc, state->mCascadeOrthoProj[i]);

			for (int j = 0; j < renderObjects.size(); j++) {
				if (renderObjects[j].materialIndex == -1 || renderObjects[j].shaderIndex == -1 || renderObjects[j].meshIndex == -1)
				{
					continue;
				}

				state->mDepthShader.setMat4(state->mDepthShaderModelLoc, renderObjects[j].model);
				Graphics::render(world, renderObjects[j], &state->mQuery);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	}
	else if (light->mLightType == LightType::Spot) {
		glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowSpotlightFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		state->mShadowProjMatrix = light->getProjMatrix();
		state->mShadowViewMatrix = glm::lookAt(lightTransform->mPosition, 
											   lightTransform->mPosition + lightTransform->getForward(), 
											   glm::vec3(0.0f, 1.0f, 0.0f));

		state->mDepthShader.use(state->mDepthShaderProgram);
		state->mDepthShader.setMat4(state->mDepthShaderProjectionLoc, state->mShadowProjMatrix);
		state->mDepthShader.setMat4(state->mDepthShaderViewLoc, state->mShadowViewMatrix);

		for (int i = 0; i < renderObjects.size(); i++) {
			if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
			{
				continue;
			}

			state->mDepthShader.setMat4(state->mDepthShaderModelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &state->mQuery);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	else if (light->mLightType == LightType::Point) {

		state->mCubeViewProjMatrices[0] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		state->mCubeViewProjMatrices[1] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		state->mCubeViewProjMatrices[2] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
		state->mCubeViewProjMatrices[3] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
		state->mCubeViewProjMatrices[4] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
		state->mCubeViewProjMatrices[5] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));

		glBindFramebuffer(GL_FRAMEBUFFER, state->mShadowCubemapFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		state->mDepthCubemapShader.use(state->mDepthCubemapShaderProgram);
		state->mDepthCubemapShader.setVec3(state->mDepthCubemapShaderLightPosLoc, lightTransform->mPosition);
		state->mDepthCubemapShader.setFloat(state->mDepthCubemapShaderFarPlaneLoc, camera->mFrustum.mFarPlane);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc0, state->mCubeViewProjMatrices[0]);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc1, state->mCubeViewProjMatrices[1]);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc2, state->mCubeViewProjMatrices[2]);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc3, state->mCubeViewProjMatrices[3]);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc4, state->mCubeViewProjMatrices[4]);
		state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc5, state->mCubeViewProjMatrices[5]);

		for (int i = 0; i < renderObjects.size(); i++) {
			if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
			{
				continue;
			}

			state->mDepthCubemapShader.setMat4(state->mDepthCubemapShaderModelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &state->mQuery);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void PhysicsEngine::renderOpaques(World* world,
	Camera* camera,
	Light* light,
	Transform* lightTransform,
	ForwardRendererState* state,
	const std::vector<RenderObject>& renderObjects)
{
	state->mLightState.mPosition = lightTransform->mPosition;
	state->mLightState.mDirection = lightTransform->getForward();
	state->mLightState.mAmbient = light->mAmbient;
	state->mLightState.mDiffuse = light->mDiffuse;
	state->mLightState.mSpecular = light->mSpecular;

	state->mLightState.mConstant = light->mConstant;
	state->mLightState.mLinear = light->mLinear;
	state->mLightState.mQuadratic = light->mQuadratic;
	state->mLightState.mCutOff = light->mCutOff;
	state->mLightState.mOuterCutOff = light->mOuterCutOff;

	if (light->mLightType == LightType::Directional) {
		for (int i = 0; i < 5; i++) {
			state->mLightState.mLightProjection[i] = state->mCascadeOrthoProj[i];

			glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, state->mCascadeEnds[i + 1], 1.0f);
			glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
			state->mLightState.mCascadeEnds[i] = clipSpaceCascadeEnd.z;

			state->mLightState.mLightView[i] = state->mCascadeLightView[i];
		}
	}
	else if (light->mLightType == LightType::Spot) {
		for (int i = 0; i < 5; i++) {
			state->mLightState.mLightProjection[i] = state->mShadowProjMatrix;
			state->mLightState.mLightView[i] = state->mShadowViewMatrix;
		}
	}

	state->mLightState.mFarPlane = camera->mFrustum.mFarPlane;

	glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &state->mLightState.mLightProjection[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &state->mLightState.mLightView[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(state->mLightState.mPosition));
	glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(state->mLightState.mDirection));
	glBufferSubData(GL_UNIFORM_BUFFER, 672, 12, glm::value_ptr(state->mLightState.mAmbient));
	glBufferSubData(GL_UNIFORM_BUFFER, 688, 12, glm::value_ptr(state->mLightState.mDiffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 704, 12, glm::value_ptr(state->mLightState.mSpecular));
	glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &state->mLightState.mCascadeEnds[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &state->mLightState.mCascadeEnds[1]);
	glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &state->mLightState.mCascadeEnds[2]);
	glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &state->mLightState.mCascadeEnds[3]);
	glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &state->mLightState.mCascadeEnds[4]);
	glBufferSubData(GL_UNIFORM_BUFFER, 800, 4, &state->mLightState.mFarPlane);
	glBufferSubData(GL_UNIFORM_BUFFER, 804, 4, &(state->mLightState.mConstant));
	glBufferSubData(GL_UNIFORM_BUFFER, 808, 4, &(state->mLightState.mLinear));
	glBufferSubData(GL_UNIFORM_BUFFER, 812, 4, &(state->mLightState.mQuadratic));
	glBufferSubData(GL_UNIFORM_BUFFER, 816, 4, &(state->mLightState.mCutOff));
	glBufferSubData(GL_UNIFORM_BUFFER, 820, 4, &(state->mLightState.mOuterCutOff));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	int variant = ShaderVariant::None;
	if (light->mLightType == LightType::Directional) {
		variant = ShaderVariant::Directional;
	}
	else if (light->mLightType == LightType::Spot) {
		variant = ShaderVariant::Spot;
	}
	else if (light->mLightType == LightType::Point) {
		variant = ShaderVariant::Point;
	}

	if (light->mShadowType == ShadowType::Hard) {
		variant |= ShaderVariant::HardShadows;
	}
	else if (light->mShadowType == ShadowType::Soft) {
		variant |= ShaderVariant::SoftShadows;
	}

	const char* const shaderShadowMapNames[] = { "shadowMap[0]",
												 "shadowMap[1]",
												 "shadowMap[2]",
												 "shadowMap[3]",
												 "shadowMap[4]" };

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());

	for (int i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);
		Shader* shader = world->getAssetByIndex<Shader>(renderObjects[i].shaderIndex);

		int shaderProgram = shader->getProgramFromVariant(variant);

		shader->use(shaderProgram);
		shader->setMat4("model", renderObjects[i].model);

		material->apply(world);

		if (light->mLightType == LightType::Directional) {
			for (int j = 0; j < 5; j++) {
				shader->setInt(shaderShadowMapNames[j], 3 + j);

				glActiveTexture(GL_TEXTURE0 + 3 + j);
				glBindTexture(GL_TEXTURE_2D, state->mShadowCascadeDepth[j]);
			}
		}
		else if (light->mLightType == LightType::Spot) {
			shader->setInt(shaderShadowMapNames[0], 3);

			glActiveTexture(GL_TEXTURE0 + 3);
			glBindTexture(GL_TEXTURE_2D, state->mShadowSpotlightDepth);
		}

		Graphics::render(world, renderObjects[i], &state->mQuery);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderColorPicking(World* world,
	Camera* camera,
	ForwardRendererState* state,
	const std::vector<RenderObject>& renderObjects)
{
	camera->clearColoring();

	// assign colors to render objects.
	int color = 1;
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 ||
			renderObjects[i].shaderIndex == -1 ||
			renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		camera->assignColoring(color, renderObjects[i].meshRendererId);

		color++;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());

	state->mColorShader.use(state->mColorShaderProgram);

	color = 1;
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 ||
			renderObjects[i].shaderIndex == -1 ||
			renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		int r = (color & 0x000000FF) >> 0;
		int g = (color & 0x0000FF00) >> 8;
		int b = (color & 0x00FF0000) >> 16;

		color++;

		state->mColorShader.setMat4(state->mColorShaderModelLoc, renderObjects[i].model);
		state->mColorShader.setVec4(state->mColorShaderColorLoc, glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f));

		Graphics::render(world, renderObjects[i], &state->mQuery);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderTransparents()
{

}

void PhysicsEngine::postProcessing()
{

}

void PhysicsEngine::endFrame(World* world,
	Camera* camera,
	ForwardRendererState* state,
	const std::vector<RenderObject>& renderObjects)
{
	glEndQuery(GL_TIME_ELAPSED);

	GLuint64 elapsedTime; // in nanoseconds
	glGetQueryObjectui64v(state->mQuery.mQueryId[state->mQuery.mQueryFront], GL_QUERY_RESULT, &elapsedTime);

	state->mQuery.mTotalElapsedTime += elapsedTime / 1000000.0f;

	// swap which query is active
	if (state->mQuery.mQueryBack) {
		state->mQuery.mQueryBack = 0;
		state->mQuery.mQueryFront = 1;
	}
	else {
		state->mQuery.mQueryBack = 1;
		state->mQuery.mQueryFront = 0;
	}

	// fill targets struct
	state->mTargets.mColor = camera->getNativeGraphicsColorTex();
	state->mTargets.mDepth = camera->getNativeGraphicsDepthTex();
	state->mTargets.mColorPicking = camera->getNativeGraphicsColorPickingTex();
	state->mTargets.mPosition = camera->getNativeGraphicsPositionTex();
	state->mTargets.mNormals = camera->getNativeGraphicsNormalTex();
	state->mTargets.mOverdraw = -1;
	state->mTargets.mSsao = camera->getNativeGraphicsSSAOColorTex();

	if (state->mRenderToScreen) {
		glViewport(0, 0, 1024, 1024);
		glScissor(0, 0, 1024, 1024);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		state->mQuadShader.use(ShaderVariant::None);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorTex());

		glBindVertexArray(state->mQuadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

void PhysicsEngine::calcShadowmapCascades(Camera* camera, ForwardRendererState* state)
{
	float nearDist = camera->mFrustum.mNearPlane;
	float farDist = camera->mFrustum.mFarPlane;

	const float splitWeight = 0.95f;
	const float ratio = farDist / nearDist;

	for (int i = 0; i < 6; i++) {
		const float si = i / 5.0f;

		state->mCascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) + (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
	}
}

void PhysicsEngine::calcCascadeOrthoProj(Camera* camera,
	glm::vec3 lightDirection,
	ForwardRendererState* state)
{
	glm::mat4 viewInv = glm::inverse(camera->getViewMatrix());
	float fov = camera->mFrustum.mFov;
	float aspect = camera->mFrustum.mAspectRatio;
	float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
	float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

	for (int i = 0; i < 5; i++) {
		float xn = -1.0f * state->mCascadeEnds[i] * tanHalfHFOV;
		float xf = -1.0f * state->mCascadeEnds[i + 1] * tanHalfHFOV;
		float yn = -1.0f * state->mCascadeEnds[i] * tanHalfVFOV;
		float yf = -1.0f * state->mCascadeEnds[i + 1] * tanHalfVFOV;

		// Find cascade frustum corners
		glm::vec4 frustumCorners[8];
		frustumCorners[0] = glm::vec4(xn, yn, state->mCascadeEnds[i], 1.0f);
		frustumCorners[1] = glm::vec4(-xn, yn, state->mCascadeEnds[i], 1.0f);
		frustumCorners[2] = glm::vec4(xn, -yn, state->mCascadeEnds[i], 1.0f);
		frustumCorners[3] = glm::vec4(-xn, -yn, state->mCascadeEnds[i], 1.0f);

		frustumCorners[4] = glm::vec4(xf, yf, state->mCascadeEnds[i + 1], 1.0f);
		frustumCorners[5] = glm::vec4(-xf, yf, state->mCascadeEnds[i + 1], 1.0f);
		frustumCorners[6] = glm::vec4(xf, -yf, state->mCascadeEnds[i + 1], 1.0f);
		frustumCorners[7] = glm::vec4(-xf, -yf, state->mCascadeEnds[i + 1], 1.0f);

		// find frustum centre by averaging corners
		glm::vec4 frustumCentre = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);

		frustumCentre += frustumCorners[0];
		frustumCentre += frustumCorners[1];
		frustumCentre += frustumCorners[2];
		frustumCentre += frustumCorners[3];
		frustumCentre += frustumCorners[4];
		frustumCentre += frustumCorners[5];
		frustumCentre += frustumCorners[6];
		frustumCentre += frustumCorners[7];

		frustumCentre *= 0.125f;
		frustumCentre.w = 1.0f;

		// Transform the frustum centre from view to world space
		glm::vec4 frustrumCentreWorldSpace = viewInv * frustumCentre;
		float d = 40.0f;//cascadeEnds[i + 1] - cascadeEnds[i];

		glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d * lightDirection.x,
			frustrumCentreWorldSpace.y + d * lightDirection.y,
			frustrumCentreWorldSpace.z + d * lightDirection.z);

		state->mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace), glm::vec3(1.0f, 0.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();

		glm::mat4 cascadeLightView = state->mCascadeLightView[i];

		// Transform the frustum coordinates from view to world space and then world to light space
		glm::vec4 vL0 = cascadeLightView * (viewInv * frustumCorners[0]);
		glm::vec4 vL1 = cascadeLightView * (viewInv * frustumCorners[1]);
		glm::vec4 vL2 = cascadeLightView * (viewInv * frustumCorners[2]);
		glm::vec4 vL3 = cascadeLightView * (viewInv * frustumCorners[3]);
		glm::vec4 vL4 = cascadeLightView * (viewInv * frustumCorners[4]);
		glm::vec4 vL5 = cascadeLightView * (viewInv * frustumCorners[5]);
		glm::vec4 vL6 = cascadeLightView * (viewInv * frustumCorners[6]);
		glm::vec4 vL7 = cascadeLightView * (viewInv * frustumCorners[7]);

		minX = glm::min(minX, vL0.x);
		maxX = glm::max(maxX, vL0.x);
		minY = glm::min(minY, vL0.y);
		maxY = glm::max(maxY, vL0.y);
		minZ = glm::min(minZ, vL0.z);
		maxZ = glm::max(maxZ, vL0.z);

		minX = glm::min(minX, vL1.x);
		maxX = glm::max(maxX, vL1.x);
		minY = glm::min(minY, vL1.y);
		maxY = glm::max(maxY, vL1.y);
		minZ = glm::min(minZ, vL1.z);
		maxZ = glm::max(maxZ, vL1.z);

		minX = glm::min(minX, vL2.x);
		maxX = glm::max(maxX, vL2.x);
		minY = glm::min(minY, vL2.y);
		maxY = glm::max(maxY, vL2.y);
		minZ = glm::min(minZ, vL2.z);
		maxZ = glm::max(maxZ, vL2.z);

		minX = glm::min(minX, vL3.x);
		maxX = glm::max(maxX, vL3.x);
		minY = glm::min(minY, vL3.y);
		maxY = glm::max(maxY, vL3.y);
		minZ = glm::min(minZ, vL3.z);
		maxZ = glm::max(maxZ, vL3.z);

		minX = glm::min(minX, vL4.x);
		maxX = glm::max(maxX, vL4.x);
		minY = glm::min(minY, vL4.y);
		maxY = glm::max(maxY, vL4.y);
		minZ = glm::min(minZ, vL4.z);
		maxZ = glm::max(maxZ, vL4.z);

		minX = glm::min(minX, vL5.x);
		maxX = glm::max(maxX, vL5.x);
		minY = glm::min(minY, vL5.y);
		maxY = glm::max(maxY, vL5.y);
		minZ = glm::min(minZ, vL5.z);
		maxZ = glm::max(maxZ, vL5.z);

		minX = glm::min(minX, vL6.x);
		maxX = glm::max(maxX, vL6.x);
		minY = glm::min(minY, vL6.y);
		maxY = glm::max(maxY, vL6.y);
		minZ = glm::min(minZ, vL6.z);
		maxZ = glm::max(maxZ, vL6.z);

		minX = glm::min(minX, vL7.x);
		maxX = glm::max(maxX, vL7.x);
		minY = glm::min(minY, vL7.y);
		maxY = glm::max(maxY, vL7.y);
		minZ = glm::min(minZ, vL7.z);
		maxZ = glm::max(maxZ, vL7.z);

		state->mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
	}
}