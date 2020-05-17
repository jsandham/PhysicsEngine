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

void PhysicsEngine::initializeRenderer(World* world, 
									   ScreenData& screenData, 
									   ShadowMapData& shadowMapData, 
									   GraphicsCameraState& cameraState, 
									   GraphicsLightState& lightState, 
									   GraphicsQuery& query)
{	
	query.mQueryBack = 0;
	query.mQueryFront = 1;

	glGenQueries(1, &(query.mQueryId[0]));
	glGenQueries(1, &(query.mQueryId[1]));

	// dummy query to prevent OpenGL errors from popping out
	//glQueryCounter(queryID[queryFrontBuffer][0], GL_TIMESTAMP);

	// generate all internal shader programs
	screenData.mPositionAndNormalsShader.setVertexShader(InternalShaders::positionAndNormalsVertexShader);
	screenData.mPositionAndNormalsShader.setFragmentShader(InternalShaders::positionAndNormalsFragmentShader);
	screenData.mPositionAndNormalsShader.compile();

	screenData.mColorShader.setVertexShader(InternalShaders::colorVertexShader);
	screenData.mColorShader.setFragmentShader(InternalShaders::colorFragmentShader);
	screenData.mColorShader.compile();

	screenData.mSsaoShader.setVertexShader(InternalShaders::ssaoVertexShader);
	screenData.mSsaoShader.setFragmentShader(InternalShaders::ssaoFragmentShader);
	screenData.mSsaoShader.compile();

	shadowMapData.mDepthShader.setVertexShader(InternalShaders::shadowDepthMapVertexShader);
	shadowMapData.mDepthShader.setFragmentShader(InternalShaders::shadowDepthMapFragmentShader);
	shadowMapData.mDepthShader.compile();

	shadowMapData.mDepthCubemapShader.setVertexShader(InternalShaders::shadowDepthCubemapVertexShader);
	shadowMapData.mDepthCubemapShader.setFragmentShader(InternalShaders::shadowDepthCubemapFragmentShader);
	shadowMapData.mDepthCubemapShader.compile();

	screenData.mQuadShader.setVertexShader(InternalShaders::windowVertexShader);
	screenData.mQuadShader.setFragmentShader(InternalShaders::windowFragmentShader);
	screenData.mQuadShader.compile();

	Graphics::checkError();

	//generate screen quad for final rendering
	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};

	glGenVertexArrays(1, &screenData.mQuadVAO);
	glBindVertexArray(screenData.mQuadVAO);

	glGenBuffers(1, &screenData.mQuadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, screenData.mQuadVBO);
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
	glGenFramebuffers(5, &shadowMapData.mShadowCascadeFBO[0]);
	glGenTextures(5, &shadowMapData.mShadowCascadeDepth[0]);

	for (int i = 0; i < 5; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowCascadeFBO[i]);
		glBindTexture(GL_TEXTURE_2D, shadowMapData.mShadowCascadeDepth[i]);
		// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapData.mShadowCascadeDepth[i], 0);

		Graphics::checkFrambufferError();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// create spotlight shadow map fbo
	glGenFramebuffers(1, &shadowMapData.mShadowSpotlightFBO);
	glGenTextures(1, &shadowMapData.mShadowSpotlightDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowSpotlightFBO);
	glBindTexture(GL_TEXTURE_2D, shadowMapData.mShadowSpotlightDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapData.mShadowSpotlightDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create pointlight shadow cubemap fbo
	glGenFramebuffers(1, &shadowMapData.mShadowCubemapFBO);
	glGenTextures(1, &shadowMapData.mShadowCubemapDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowCubemapFBO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMapData.mShadowCubemapDepth);
	for (unsigned int i = 0; i < 6; i++) {
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	}

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowMapData.mShadowCubemapDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	glGenBuffers(1, &(cameraState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.mHandle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(lightState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, lightState.mHandle);
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

			/*int count = renderObjects.size() / 8;

			int b = count % (256 * 256);
			count = count - b * 256 * 256;
			int g = count % 256;
			count = count - g * 256;
			int r = count;

			Color pickingColor = Color(r, g, b, 255);*/

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

				//renderObject.color = pickingColor;

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

		//Log::info(("centre before: " + std::to_string(centre.x) + " " + std::to_string(centre.y) + " " + std::to_string(centre.z) + " " + std::to_string(radius) + "\n").c_str());

		//glm::vec4 test = camera->getViewMatrix() * renderObjects[i].model * glm::vec4(centre.x, centre.y, centre.z, 1.0f);
		glm::vec4 test = renderObjects[i].model * glm::vec4(centre.x, centre.y, centre.z, 1.0f);

		//Log::info(("centre after: " + std::to_string(test.x) + " " + std::to_string(test.y) + " " + std::to_string(test.z) + "\n").c_str());

		Sphere cullingSphere;
		cullingSphere.mCentre = glm::vec3(test.x, test.y, test.z);
		cullingSphere.mRadius = radius;

		if (Geometry::intersect(cullingSphere, camera->mFrustum)) {
			count++;
		}
	}

	//Log::info(("count: " + std::to_string(count) + "\n").c_str());
}

void PhysicsEngine::updateModelMatrices(World* world, std::vector<RenderObject>& renderObjects)
{
	const PoolAllocator<Transform>* transformAllocator = world->getComponentAllocator_Const<Transform>();

	// update model matrices
	int n = (int)renderObjects.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		Transform* transform = world->getComponentByIndex<Transform>(transformAllocator, renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();
	}
}

void PhysicsEngine::beginFrame(World* world, 
							   Camera* camera, 
							   GraphicsCameraState& cameraState, 
							   GraphicsLightState& lightState, 
							   GraphicsQuery& query)
{
	query.mNumBatchDrawCalls = 0;
	query.mNumDrawCalls = 0;
	query.mTotalElapsedTime = 0.0f;
	query.mVerts = 0;
	query.mTris = 0;
	query.mLines = 0;
	query.mPoints = 0;

	LOG_OGL(glBeginQuery(GL_TIME_ELAPSED, query.mQueryId[query.mQueryBack]);)

	cameraState.mProjection = camera->getProjMatrix();
	cameraState.mView = camera->getViewMatrix();
	cameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

	// set camera state binding point and update camera state data
	LOG_OGL(glBindBuffer(GL_UNIFORM_BUFFER, cameraState.mHandle);)
	LOG_OGL(glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraState.mHandle, 0, 144);)
	LOG_OGL(glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(cameraState.mProjection));)
	LOG_OGL(glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(cameraState.mView));)
	LOG_OGL(glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(cameraState.mCameraPos));)
	LOG_OGL(glBindBuffer(GL_UNIFORM_BUFFER, 0);)

	// set light state binding point
	LOG_OGL(glBindBuffer(GL_UNIFORM_BUFFER, lightState.mHandle);)
	LOG_OGL(glBindBufferRange(GL_UNIFORM_BUFFER, 1, lightState.mHandle, 0, 824);)
	LOG_OGL(glBindBuffer(GL_UNIFORM_BUFFER, 0);)

	LOG_OGL(glEnable(GL_SCISSOR_TEST);)
	LOG_OGL(glEnable(GL_DEPTH_TEST);)
	LOG_OGL(glEnable(GL_BLEND);)
	LOG_OGL(glDepthFunc(GL_LEQUAL);)
	LOG_OGL(glBlendFunc(GL_ONE, GL_ZERO);)
	LOG_OGL(glBlendEquation(GL_FUNC_ADD);)

	LOG_OGL(glViewport(camera->mViewport.mX, camera->mViewport.mY, camera->mViewport.mWidth, camera->mViewport.mHeight);)
	LOG_OGL(glScissor(camera->mViewport.mX, camera->mViewport.mY, camera->mViewport.mWidth, camera->mViewport.mHeight);)

	LOG_OGL(glClearColor(camera->mBackgroundColor.x, camera->mBackgroundColor.y, camera->mBackgroundColor.z, camera->mBackgroundColor.w);)
	LOG_OGL(glClearDepth(1.0f);)

	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());)
	LOG_OGL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);)
	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, 0);)

	LOG_OGL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f);)

	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());)
	LOG_OGL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);)
	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, 0);)

	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());)
	LOG_OGL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);)
	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, 0);)

	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());)
	LOG_OGL(glClear(GL_COLOR_BUFFER_BIT);)
	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, 0);)

	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());)
	LOG_OGL(glClear(GL_COLOR_BUFFER_BIT);)
	LOG_OGL(glBindFramebuffer(GL_FRAMEBUFFER, 0);)

}

void PhysicsEngine::computeSSAO(World* world, 
								Camera* camera, 
								const std::vector<RenderObject>& renderObjects, 
								ScreenData& screenData, 
								GraphicsQuery& query)
{
	// fill geometry framebuffer
	int shaderProgram = screenData.mPositionAndNormalsShader.getProgramFromVariant(ShaderVariant::None);
	int modelLoc = screenData.mPositionAndNormalsShader.findUniformLocation("model", shaderProgram);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		screenData.mPositionAndNormalsShader.use(shaderProgram);
		screenData.mPositionAndNormalsShader.setMat4(modelLoc, renderObjects[i].model);

		Graphics::render(world, renderObjects[i], &query);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	// fill ssao color texture
	shaderProgram = screenData.mSsaoShader.getProgramFromVariant(ShaderVariant::None);
	int projectionLoc = screenData.mSsaoShader.findUniformLocation("projection", shaderProgram);
	int positionTexLoc = screenData.mSsaoShader.findUniformLocation("positionTex", shaderProgram);
	int normalTexLoc = screenData.mSsaoShader.findUniformLocation("normalTex", shaderProgram);
	int noiseTexLoc = screenData.mSsaoShader.findUniformLocation("noiseTex", shaderProgram);
	int samplesLoc[64];
	for (int i = 0; i < 64; i++) {
		samplesLoc[i] = screenData.mSsaoShader.findUniformLocation("samples[" + std::to_string(i) + "]", shaderProgram);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());
	screenData.mSsaoShader.use(shaderProgram);
	screenData.mSsaoShader.setMat4(projectionLoc, camera->getProjMatrix());
	for (int i = 0; i < 64; i++) {
		screenData.mSsaoShader.setVec3(samplesLoc[i], camera->getSSAOSample(i));
	}
	screenData.mSsaoShader.setInt(positionTexLoc, 0);
	screenData.mSsaoShader.setInt(normalTexLoc, 1);
	screenData.mSsaoShader.setInt(noiseTexLoc, 2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsPositionTex());
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsNormalTex());
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsSSAONoiseTex());

	glBindVertexArray(screenData.mQuadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderShadows(World* world, 
								  Camera* camera, 
								  Light* light, 
								  Transform* lightTransform,
								  const std::vector<RenderObject>& renderObjects, 
								  ShadowMapData& shadowMapData, 
								  GraphicsQuery& query)
{
	LightType lightType = light->mLightType;

	if (lightType == LightType::Directional) {

		calcShadowmapCascades(camera, shadowMapData);
		calcCascadeOrthoProj(camera, lightTransform->getForward(), shadowMapData);

		int shaderProgram = shadowMapData.mDepthShader.getProgramFromVariant(ShaderVariant::None);
		int modelLoc = shadowMapData.mDepthShader.findUniformLocation("model", shaderProgram);
		int viewLoc = shadowMapData.mDepthShader.findUniformLocation("view", shaderProgram);
		int projectionLoc = shadowMapData.mDepthShader.findUniformLocation("projection", shaderProgram);

		for (int i = 0; i < 5; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowCascadeFBO[i]);

			glClearDepth(1.0f);
			glClear(GL_DEPTH_BUFFER_BIT);

			shadowMapData.mDepthShader.use(shaderProgram);
			shadowMapData.mDepthShader.setMat4(viewLoc, shadowMapData.mCascadeLightView[i]);
			shadowMapData.mDepthShader.setMat4(projectionLoc, shadowMapData.mCascadeOrthoProj[i]);

			for (int j = 0; j < renderObjects.size(); j++) {
				if (renderObjects[j].materialIndex == -1 || renderObjects[j].shaderIndex == -1 || renderObjects[j].meshIndex == -1)
				{
					continue;
				}

				shadowMapData.mDepthShader.setMat4(modelLoc, renderObjects[j].model);
				Graphics::render(world, renderObjects[j], &query);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	}
	else if (lightType == LightType::Spot) {

		int shaderProgram = shadowMapData.mDepthShader.getProgramFromVariant(ShaderVariant::None);
		int modelLoc = shadowMapData.mDepthShader.findUniformLocation("model", shaderProgram);
		int viewLoc = shadowMapData.mDepthShader.findUniformLocation("view", shaderProgram);
		int projectionLoc = shadowMapData.mDepthShader.findUniformLocation("projection", shaderProgram);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowSpotlightFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadowMapData.mShadowProjMatrix = light->getProjMatrix();
		shadowMapData.mShadowViewMatrix = glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + lightTransform->getForward(), glm::vec3(0.0f, 1.0f, 0.0f));

		shadowMapData.mDepthShader.use(shaderProgram);
		shadowMapData.mDepthShader.setMat4(projectionLoc, shadowMapData.mShadowProjMatrix);
		shadowMapData.mDepthShader.setMat4(viewLoc, shadowMapData.mShadowViewMatrix);

		for (int i = 0; i < renderObjects.size(); i++) {
			if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
			{
				continue;
			}

			shadowMapData.mDepthShader.setMat4(modelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	else if (lightType == LightType::Point) {

		shadowMapData.mCubeViewProjMatrices[0] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.mCubeViewProjMatrices[1] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.mCubeViewProjMatrices[2] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
		shadowMapData.mCubeViewProjMatrices[3] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
		shadowMapData.mCubeViewProjMatrices[4] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.mCubeViewProjMatrices[5] = (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));

		int shaderProgram = shadowMapData.mDepthCubemapShader.getProgramFromVariant(ShaderVariant::None);
		int lightPosLoc = shadowMapData.mDepthCubemapShader.findUniformLocation("lightPos", shaderProgram);
		int farPlaneLoc = shadowMapData.mDepthCubemapShader.findUniformLocation("farPlane", shaderProgram);
		int modelLoc = shadowMapData.mDepthCubemapShader.findUniformLocation("model", shaderProgram);
		int cubeViewProjMatricesLoc0 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[0]", shaderProgram);
		int cubeViewProjMatricesLoc1 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[1]", shaderProgram);
		int cubeViewProjMatricesLoc2 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[2]", shaderProgram);
		int cubeViewProjMatricesLoc3 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[3]", shaderProgram);
		int cubeViewProjMatricesLoc4 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[4]", shaderProgram);
		int cubeViewProjMatricesLoc5 = shadowMapData.mDepthCubemapShader.findUniformLocation("cubeViewProjMatrices[5]", shaderProgram);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.mShadowCubemapFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadowMapData.mDepthCubemapShader.use(shaderProgram);
		shadowMapData.mDepthCubemapShader.setVec3(lightPosLoc, lightTransform->mPosition);
		shadowMapData.mDepthCubemapShader.setFloat(farPlaneLoc, camera->mFrustum.mFarPlane);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc0, shadowMapData.mCubeViewProjMatrices[0]);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc1, shadowMapData.mCubeViewProjMatrices[1]);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc2, shadowMapData.mCubeViewProjMatrices[2]);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc3, shadowMapData.mCubeViewProjMatrices[3]);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc4, shadowMapData.mCubeViewProjMatrices[4]);
		shadowMapData.mDepthCubemapShader.setMat4(cubeViewProjMatricesLoc5, shadowMapData.mCubeViewProjMatrices[5]);

		for (int i = 0; i < renderObjects.size(); i++) {
			if (renderObjects[i].materialIndex == -1 || renderObjects[i].shaderIndex == -1 || renderObjects[i].meshIndex == -1)
			{
				continue;
			}

			shadowMapData.mDepthCubemapShader.setMat4(modelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void PhysicsEngine::renderOpaques(World* world, 
								  Camera* camera, 
								  Light* light, 
								  Transform* lightTransform, 
								  const std::vector<RenderObject>& renderObjects, 
								  const ShadowMapData& shadowMapData, 
								  GraphicsLightState& lightState, 
								  GraphicsQuery& query)
{
	lightState.mPosition = lightTransform->mPosition;
	lightState.mDirection = lightTransform->getForward();
	lightState.mAmbient = light->mAmbient;
	lightState.mDiffuse = light->mDiffuse;
	lightState.mSpecular = light->mSpecular;

	lightState.mConstant = light->mConstant;
	lightState.mLinear = light->mLinear;
	lightState.mQuadratic = light->mQuadratic;
	lightState.mCutOff = light->mCutOff;
	lightState.mOuterCutOff = light->mOuterCutOff;

	if (light->mLightType == LightType::Directional) {
		for (int i = 0; i < 5; i++) {
			lightState.mLightProjection[i] = shadowMapData.mCascadeOrthoProj[i];

			glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, shadowMapData.mCascadeEnds[i + 1], 1.0f);
			glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
			lightState.mCascadeEnds[i] = clipSpaceCascadeEnd.z;

			lightState.mLightView[i] = shadowMapData.mCascadeLightView[i];
		}
	}
	else if (light->mLightType == LightType::Spot) {
		for (int i = 0; i < 5; i++) {
			lightState.mLightProjection[i] = shadowMapData.mShadowProjMatrix;
			lightState.mLightView[i] = shadowMapData.mShadowViewMatrix;
		}
	}

	lightState.mFarPlane = camera->mFrustum.mFarPlane;

	glBindBuffer(GL_UNIFORM_BUFFER, lightState.mHandle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &lightState.mLightProjection[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &lightState.mLightView[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(lightState.mPosition));
	glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(lightState.mDirection));
	glBufferSubData(GL_UNIFORM_BUFFER, 672, 12, glm::value_ptr(lightState.mAmbient));
	glBufferSubData(GL_UNIFORM_BUFFER, 688, 12, glm::value_ptr(lightState.mDiffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 704, 12, glm::value_ptr(lightState.mSpecular));
	glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &lightState.mCascadeEnds[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &lightState.mCascadeEnds[1]);
	glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &lightState.mCascadeEnds[2]);
	glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &lightState.mCascadeEnds[3]);
	glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &lightState.mCascadeEnds[4]);
	glBufferSubData(GL_UNIFORM_BUFFER, 800, 4, &lightState.mFarPlane);
	glBufferSubData(GL_UNIFORM_BUFFER, 804, 4, &(lightState.mConstant));
	glBufferSubData(GL_UNIFORM_BUFFER, 808, 4, &(lightState.mLinear));
	glBufferSubData(GL_UNIFORM_BUFFER, 812, 4, &(lightState.mQuadratic));
	glBufferSubData(GL_UNIFORM_BUFFER, 816, 4, &(lightState.mCutOff));
	glBufferSubData(GL_UNIFORM_BUFFER, 820, 4, &(lightState.mOuterCutOff));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	int variant = ShaderVariant::None;
	if (light->mLightType == LightType::Directional) {
		variant = ShaderVariant::Directional;
	}
	else if (light->mLightType == LightType::Spot) {
		variant = ShaderVariant::Spot;
	}
	else if(light->mLightType == LightType::Point) {
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
				glBindTexture(GL_TEXTURE_2D, shadowMapData.mShadowCascadeDepth[j]);
			}
		}
		else if (light->mLightType == LightType::Spot) {
			shader->setInt(shaderShadowMapNames[0], 3);

			glActiveTexture(GL_TEXTURE0 + 3);
			glBindTexture(GL_TEXTURE_2D, shadowMapData.mShadowSpotlightDepth);
		}

		Graphics::render(world, renderObjects[i], &query);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderColorPicking(World* world,
									   Camera* camera,
									   const std::vector<RenderObject>& renderObjects,
									   ScreenData& screenData,
									   GraphicsQuery& query)
{
	camera->clearColoring();

	// assign colors to render objects.
	size_t color = 1;
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

	int shaderProgram = screenData.mColorShader.getProgramFromVariant(ShaderVariant::None);
	int modelLoc = screenData.mColorShader.findUniformLocation("model", shaderProgram);
	int colorLoc = screenData.mColorShader.findUniformLocation("color", shaderProgram);

	screenData.mColorShader.use(shaderProgram);

	color = 1;
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (renderObjects[i].materialIndex == -1 || 
			renderObjects[i].shaderIndex == -1 || 
			renderObjects[i].meshIndex == -1)
		{
			continue;
		}

		size_t b = color / (256 * 256);
		color = color % (256 * 256);
		size_t g = color / 256;
		color = color % 256;
		size_t r = color;

		color++; 

		screenData.mColorShader.setMat4(modelLoc, renderObjects[i].model);
		screenData.mColorShader.setVec4(colorLoc, glm::vec4(r / 256.0f, g / 256.0f, b / 256.0f, 1.0f));

		Graphics::render(world, renderObjects[i], &query);
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
							 const std::vector<RenderObject>& renderObjects, 
							 ScreenData& screenData, 
							 GraphicsTargets& targets, 
							 GraphicsQuery& query, 
							 bool renderToScreen)
{
	glEndQuery(GL_TIME_ELAPSED);

	GLuint64 elapsedTime; // in nanoseconds
	glGetQueryObjectui64v(query.mQueryId[query.mQueryFront], GL_QUERY_RESULT, &elapsedTime);

	query.mTotalElapsedTime += elapsedTime / 1000000.0f;

	// swap which query is active
	if (query.mQueryBack) {
		query.mQueryBack = 0;
		query.mQueryFront = 1;
	}
	else {
		query.mQueryBack = 1;
		query.mQueryFront = 0;
	}

	/*if (world->debug) {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		int view = world->debugView;

		glBindFramebuffer(GL_FRAMEBUFFER, debug.fbo[view].handle);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(view == 0 || view == 1 || view == 2){ 

			GLuint shaderProgram = debug.shaders[view].programs[(int)ShaderVariant::None].handle;

			Graphics::use(shaderProgram);

			for(int i = 0; i < renderObjects.size(); i++){
				Graphics::setMat4(shaderProgram, "model", renderObjects[i].model);
				Graphics::render(world, renderObjects[i], &query);
			}
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}*/

	// fill targets struct
	targets.mColor = camera->getNativeGraphicsColorTex();
	targets.mDepth = camera->getNativeGraphicsDepthTex();
	targets.mColorPicking = camera->getNativeGraphicsColorPickingTex();
	targets.mPosition = camera->getNativeGraphicsPositionTex();
	targets.mNormals = camera->getNativeGraphicsNormalTex();
	targets.mOverdraw = -1;
	targets.mSsao = camera->getNativeGraphicsSSAOColorTex();
	/*if (world->debug) {
		targets.depth = debug.fbo[0].depthBuffer.handle.handle;
		targets.normals = debug.fbo[1].colorBuffer.handle.handle;
		targets.overdraw = debug.fbo[2].colorBuffer.handle.handle;
	}*/

	// choose current target
	//GLuint drawTex = camera->colorTex;
	//if (world->debug) {
	//	int view = world->debugView;
	//	if (view == 0) {
	//		drawTex = camera->depthTex;
	//		//drawTex = debug.fbo[view].depthBuffer.handle.handle;
	//	}
	//	else {
	//		drawTex = camera->normalTex;
	//		//drawTex = debug.fbo[view].colorBuffer.handle.handle;
	//	}
	//}

	if (renderToScreen) {
		glViewport(0, 0, 1024, 1024);
		glScissor(0, 0, 1024, 1024);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		screenData.mQuadShader.use(ShaderVariant::None);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorTex());

		glBindVertexArray(screenData.mQuadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

void PhysicsEngine::calcShadowmapCascades(Camera* camera, ShadowMapData& shadowMapData)
{
	float nearDist = camera->mFrustum.mNearPlane;
	float farDist = camera->mFrustum.mFarPlane;

	const float splitWeight = 0.95f;
	const float ratio = farDist / nearDist;

	for (int i = 0; i < 6; i++) {
		const float si = i / 5.0f;

		shadowMapData.mCascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) + (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
	}
}

void PhysicsEngine::calcCascadeOrthoProj(Camera* camera, 
										 glm::vec3 lightDirection, 
										 ShadowMapData& shadowMapData)
{
	glm::mat4 viewInv = glm::inverse(camera->getViewMatrix());
	float fov = camera->mFrustum.mFov;
	float aspect = camera->mFrustum.mAspectRatio;
	float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
	float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

	for (unsigned int i = 0; i < 5; i++) {
		float xn = -1.0f * shadowMapData.mCascadeEnds[i] * tanHalfHFOV;
		float xf = -1.0f * shadowMapData.mCascadeEnds[i + 1] * tanHalfHFOV;
		float yn = -1.0f * shadowMapData.mCascadeEnds[i] * tanHalfVFOV;
		float yf = -1.0f * shadowMapData.mCascadeEnds[i + 1] * tanHalfVFOV;

		glm::vec4 frustumCorners[8];
		frustumCorners[0] = glm::vec4(xn, yn, shadowMapData.mCascadeEnds[i], 1.0f);
		frustumCorners[1] = glm::vec4(-xn, yn, shadowMapData.mCascadeEnds[i], 1.0f);
		frustumCorners[2] = glm::vec4(xn, -yn, shadowMapData.mCascadeEnds[i], 1.0f);
		frustumCorners[3] = glm::vec4(-xn, -yn, shadowMapData.mCascadeEnds[i], 1.0f);

		frustumCorners[4] = glm::vec4(xf, yf, shadowMapData.mCascadeEnds[i + 1], 1.0f);
		frustumCorners[5] = glm::vec4(-xf, yf, shadowMapData.mCascadeEnds[i + 1], 1.0f);
		frustumCorners[6] = glm::vec4(xf, -yf, shadowMapData.mCascadeEnds[i + 1], 1.0f);
		frustumCorners[7] = glm::vec4(-xf, -yf, shadowMapData.mCascadeEnds[i + 1], 1.0f);

		glm::vec4 frustumCentre = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		for (int j = 0; j < 8; j++) {
			frustumCentre.x += frustumCorners[j].x;
			frustumCentre.y += frustumCorners[j].y;
			frustumCentre.z += frustumCorners[j].z;
		}

		frustumCentre.x = frustumCentre.x / 8;
		frustumCentre.y = frustumCentre.y / 8;
		frustumCentre.z = frustumCentre.z / 8;

		// Transform the frustum centre from view to world space
		glm::vec4 frustrumCentreWorldSpace = viewInv * frustumCentre;
		float d = 40.0f;//cascadeEnds[i + 1] - cascadeEnds[i];

		glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d * lightDirection.x, 
								frustrumCentreWorldSpace.y + d * lightDirection.y, 
								frustrumCentreWorldSpace.z + d * lightDirection.z);

		shadowMapData.mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace.x, frustrumCentreWorldSpace.y, frustrumCentreWorldSpace.z), glm::vec3(1.0f, 0.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();
		for (unsigned int j = 0; j < 8; j++) {
			// Transform the frustum coordinate from view to world space
			glm::vec4 vW = viewInv * frustumCorners[j];

			// Transform the frustum coordinate from world to light space
			glm::vec4 vL = shadowMapData.mCascadeLightView[i] * vW;

			minX = glm::min(minX, vL.x);
			maxX = glm::max(maxX, vL.x);
			minY = glm::min(minY, vL.y);
			maxY = glm::max(maxY, vL.y);
			minZ = glm::min(minZ, vL.z);
			maxZ = glm::max(maxZ, vL.z);
		}

		shadowMapData.mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
	}
}