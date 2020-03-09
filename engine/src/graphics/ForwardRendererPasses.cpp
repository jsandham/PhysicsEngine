#include <random>
#include <unordered_set>

#include "../../include/core/Shader.h"
#include "../../include/core/InternalShaders.h"

#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

void PhysicsEngine::initializeRenderer(World* world, ScreenData& screenData, ShadowMapData& shadowMapData, GraphicsCameraState& cameraState, GraphicsLightState& lightState, GraphicsDebug& debug, GraphicsQuery& query)
{
	glGenQueries(1, &(query.queryId));

	// generate all internal shader programs
	screenData.positionAndNormalsShader.setVertexShader(InternalShaders::positionAndNormalsVertexShader);
	screenData.positionAndNormalsShader.setFragmentShader(InternalShaders::positionAndNormalsFragmentShader);
	screenData.positionAndNormalsShader.compile();

	screenData.ssaoShader.setVertexShader(InternalShaders::ssaoVertexShader);
	screenData.ssaoShader.setFragmentShader(InternalShaders::ssaoFragmentShader);
	screenData.ssaoShader.compile();

	shadowMapData.depthShader.setVertexShader(InternalShaders::shadowDepthMapVertexShader);
	shadowMapData.depthShader.setFragmentShader(InternalShaders::shadowDepthMapFragmentShader);
	shadowMapData.depthShader.compile();

	shadowMapData.depthCubemapShader.setVertexShader(InternalShaders::shadowDepthCubemapVertexShader);
	shadowMapData.depthCubemapShader.setFragmentShader(InternalShaders::shadowDepthCubemapFragmentShader);
	shadowMapData.depthCubemapShader.compile();

	screenData.quadShader.setVertexShader(InternalShaders::windowVertexShader);
	screenData.quadShader.setFragmentShader(InternalShaders::windowFragmentShader);
	screenData.quadShader.compile();

	Graphics::checkError();

	//generate screen quad for final rendering
	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};

	glGenVertexArrays(1, &screenData.quadVAO);
	glBindVertexArray(screenData.quadVAO);

	glGenBuffers(1, &screenData.quadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, screenData.quadVBO);
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
	glGenFramebuffers(5, &shadowMapData.shadowCascadeFBO[0]);
	glGenTextures(5, &shadowMapData.shadowCascadeDepth[0]);

	for (int i = 0; i < 5; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowCascadeFBO[i]);
		glBindTexture(GL_TEXTURE_2D, shadowMapData.shadowCascadeDepth[i]);
		// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapData.shadowCascadeDepth[i], 0);

		Graphics::checkFrambufferError();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// create spotlight shadow map fbo
	glGenFramebuffers(1, &shadowMapData.shadowSpotlightFBO);
	glGenTextures(1, &shadowMapData.shadowSpotlightDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowSpotlightFBO);
	glBindTexture(GL_TEXTURE_2D, shadowMapData.shadowSpotlightDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapData.shadowSpotlightDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create pointlight shadow cubemap fbo
	glGenFramebuffers(1, &shadowMapData.shadowCubemapFBO);
	glGenTextures(1, &shadowMapData.shadowCubemapDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowCubemapFBO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMapData.shadowCubemapDepth);
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

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowMapData.shadowCubemapDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	glGenBuffers(1, &(cameraState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraState.handle, 0, 144);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(lightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, lightState.handle, 0, 824);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	if (world->debug) {
		debug.init();
	}

	Graphics::checkError();
}

void PhysicsEngine::registerRenderAssets(World* world)
{
	// create all texture assets not already created
	for (int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++) {
		Texture2D* texture = world->getAssetByIndex<Texture2D>(i);
		if (texture != NULL && !texture->isCreated) {
			int width = texture->getWidth();
			int height = texture->getHeight();
			int numChannels = texture->getNumChannels();
			TextureFormat format = texture->getFormat();
			std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

			glGenTextures(1, &(texture->tex));
			glBindTexture(GL_TEXTURE_2D, texture->tex);

			GLenum openglFormat = GL_DEPTH_COMPONENT;
			switch (format)
			{
			case Depth:
				openglFormat = GL_DEPTH_COMPONENT;
				break;
			case RG:
				openglFormat = GL_RG;
				break;
			case RGB:
				openglFormat = GL_RGB;
				break;
			case RGBA:
				openglFormat = GL_RGBA;
				break;
			default:
				Log::error("Invalid texture format\n");
			}

			glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

			glGenerateMipmap(GL_TEXTURE_2D);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			glBindTexture(GL_TEXTURE_2D, 0);

			texture->isCreated = true;
		}
	}

	Graphics::checkError();

	// compile all shader assets and configure uniform blocks not already compiled
	std::unordered_set<Guid> shadersCompiledThisFrame;
	for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
		Shader* shader = world->getAssetByIndex<Shader>(i);

		if (!shader->isCompiled()) {

			shader->compile();

			shadersCompiledThisFrame.insert(shader->getId());

			if (!shader->isCompiled()) {
				std::string errorMessage = "Shader failed to compile " + shader->getId().toString() + "\n";
				Log::error(&errorMessage[0]);
			}

			shader->setUniformBlock("CamerBlock", 0);
			shader->setUniformBlock("LightBlock", 1);
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

		if (mesh != NULL && !mesh->isCreated) {
			glGenVertexArrays(1, &mesh->vao);
			glBindVertexArray(mesh->vao);
			glGenBuffers(1, &mesh->vbo[0]);
			glGenBuffers(1, &mesh->vbo[1]);
			glGenBuffers(1, &mesh->vbo[2]);

			glBindVertexArray(mesh->vao);
			glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, mesh->getVertices().size() * sizeof(float), &(mesh->getVertices()[0]), GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo[1]);
			glBufferData(GL_ARRAY_BUFFER, mesh->getNormals().size() * sizeof(float), &(mesh->getNormals()[0]), GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo[2]);
			glBufferData(GL_ARRAY_BUFFER, mesh->getTexCoords().size() * sizeof(float), &(mesh->getTexCoords()[0]), GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

			glBindVertexArray(0);

			mesh->isCreated = true;
		}
	}

	Graphics::checkError();
}

void PhysicsEngine::registerRenderObjects(World* world, std::vector<RenderObject>& renderObjects)
{
	const int meshRendererType = ComponentType<MeshRenderer>::type;
	const int transformType = ComponentType<Transform>::type;

	// add created mesh renderers to render object list
	std::vector<Guid> meshRendererIdsAdded;

	std::vector<triple<Guid, Guid, int>> componentIdsAdded = world->getComponentIdsMarkedCreated();
	for (size_t i = 0; i < componentIdsAdded.size(); i++) {
		if (componentIdsAdded[i].third == meshRendererType) {
			meshRendererIdsAdded.push_back(componentIdsAdded[i].second);
		}
	}

	for (size_t i = 0; i < meshRendererIdsAdded.size(); i++) {
		int globalIndex = world->getIndexOf(meshRendererIdsAdded[i]);

		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(globalIndex);
		if (meshRenderer != NULL && !meshRenderer->isStatic) {
			addToRenderObjectsList(world, meshRenderer, renderObjects);
		}
	}

	// remove destroyed mesh renderers from render objects list
	std::vector<Guid> meshRendererIdsDestroyed;

	std::vector<triple<Guid, Guid, int>> componentIdsDestroyed = world->getComponentIdsMarkedLatentDestroy();
	for (size_t i = 0; i < componentIdsDestroyed.size(); i++) {
		if (componentIdsDestroyed[i].third == meshRendererType) {
			meshRendererIdsDestroyed.push_back(componentIdsDestroyed[i].second);
		}
	}

	for (size_t i = 0; i < meshRendererIdsDestroyed.size(); i++) {
		int globalIndex = world->getIndexOf(meshRendererIdsDestroyed[i]);

		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(globalIndex);
		if (meshRenderer != NULL && !meshRenderer->isStatic) {
			removeFromRenderObjectsList(meshRenderer, renderObjects);
		}
	}

	// update mesh renderers that have been moved in their global arrays
	std::vector<std::pair<int, int>> transformIndicesMoved;

	std::vector<triple<Guid, int, int>> componentIdsMoved = world->getComponentIdsMarkedMoved();
	for (size_t i = 0; i < componentIdsMoved.size(); i++) {
		if (componentIdsMoved[i].second == transformType) {
			int oldIndex = componentIdsMoved[i].third;
			int newIndex = world->getIndexOf(componentIdsMoved[i].first);

			transformIndicesMoved.push_back(std::make_pair(oldIndex, newIndex));
		}
	}

	for (size_t i = 0; i < transformIndicesMoved.size(); i++) {
		for (size_t j = 0; j < renderObjects.size(); j++) {
			if (transformIndicesMoved[i].first == renderObjects[j].transformIndex) {
				renderObjects[j].transformIndex = transformIndicesMoved[i].second;
			}
		}
	}
}

void PhysicsEngine::registerCameras(World* world)
{
	for (int i = 0; i < world->getNumberOfComponents<Camera>(); i++) {
		Camera* camera = world->getComponentByIndex<Camera>(i);

		if (!camera->isCreated) {
			// generate main camera fbo (color + depth)
			glGenFramebuffers(1, &camera->mainFBO);
			glBindFramebuffer(GL_FRAMEBUFFER, camera->mainFBO);

			glGenTextures(1, &camera->colorTex);
			glBindTexture(GL_TEXTURE_2D, camera->colorTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glGenTextures(1, &camera->depthTex);
			glBindTexture(GL_TEXTURE_2D, camera->depthTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, camera->colorTex, 0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, camera->depthTex, 0);

			// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
			unsigned int mainAttachments[1] = { GL_COLOR_ATTACHMENT0 };
			glDrawBuffers(1, mainAttachments);

			Graphics::checkFrambufferError();

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// generate geometry fbo
			glGenFramebuffers(1, &camera->geometryFBO);
			glBindFramebuffer(GL_FRAMEBUFFER, camera->geometryFBO);

			glGenTextures(1, &camera->positionTex);
			glBindTexture(GL_TEXTURE_2D, camera->positionTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glGenTextures(1, &camera->normalTex);
			glBindTexture(GL_TEXTURE_2D, camera->normalTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, camera->positionTex, 0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, camera->normalTex, 0);

			unsigned int geometryAttachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
			glDrawBuffers(2, geometryAttachments);

			Graphics::checkFrambufferError();

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// generate ssao fbo
			glGenFramebuffers(1, &camera->ssaoFBO);
			glBindFramebuffer(GL_FRAMEBUFFER, camera->ssaoFBO);

			glGenTextures(1, &camera->ssaoColorTex);
			glBindTexture(GL_TEXTURE_2D, camera->ssaoColorTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, camera->ssaoColorTex, 0);
			
			unsigned int ssaoAttachments[1] = { GL_COLOR_ATTACHMENT0 };
			glDrawBuffers(1, ssaoAttachments);

			Graphics::checkFrambufferError();

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			auto lerp = [](float a, float b, float t) { return a + t * (b - a); };

			//generate noise texture for use in ssao
			std::uniform_real_distribution<GLfloat> distribution(0.0, 1.0);
			std::default_random_engine generator;
			for (unsigned int j = 0; j < 64; ++j)
			{
				float x = distribution(generator) * 2.0f - 1.0f;
				float y = distribution(generator) * 2.0f - 1.0f;
				float z = distribution(generator);
				float radius = distribution(generator);

				glm::vec3 sample(x, y, z);
				sample = radius * glm::normalize(sample);
				float scale = float(j) / 64.0f;

				// scale samples s.t. they're more aligned to center of kernel
				scale = lerp(0.1f, 1.0f, scale * scale);
				sample *= scale;
				camera->ssaoSamples.push_back(sample);
			}

			std::vector<glm::vec3> ssaoNoise;
			for (int j = 0; j < 16; j++) {
				// rotate around z-axis (in tangent space)
				glm::vec3 noise(distribution(generator) * 2.0f - 1.0f, distribution(generator) * 2.0f - 1.0f, 0.0f);
				ssaoNoise.push_back(noise);
			}

			glGenTextures(1, &camera->ssaoNoiseTex);
			glBindTexture(GL_TEXTURE_2D, camera->ssaoNoiseTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			Graphics::checkError();

			camera->isCreated = true;
		}
	}
}

void PhysicsEngine::cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects)
{

}

void PhysicsEngine::updateTransforms(World* world, std::vector<RenderObject>& renderObjects)
{
	// update model matrices
	int n = (int)renderObjects.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++) {
		Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();

		/*glm::vec4 temp = renderObjects[i].model * glm::vec4(renderObjects[i].boundingSphere.centre, 1.0f);
		glm::vec3 centre = glm::vec3(temp.x, temp.y, temp.z);
		float radius = renderObjects[i].boundingSphere.radius;*/

		//if(camera->checkSphereInFrustum(centre, radius)){
		//	std::cout << "Render object inside camera frustrum " << centre.x << " " << centre.y << " " << centre.z << " " << radius << std::endl;
		//}
	}
}

void PhysicsEngine::beginFrame(Camera* camera, GraphicsCameraState& cameraState, GraphicsQuery& query)
{
	query.numBatchDrawCalls = 0;
	query.numDrawCalls = 0;
	query.totalElapsedTime = 0.0f;
	query.verts = 0;
	query.tris = 0;
	query.lines = 0;
	query.points = 0;

	cameraState.projection = camera->getProjMatrix();
	cameraState.view = camera->getViewMatrix();
	cameraState.cameraPos = camera->position;

	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(cameraState.projection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(cameraState.view));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(cameraState.cameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glEnable(GL_SCISSOR_TEST);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);

	glViewport(camera->viewport.x, camera->viewport.y, camera->viewport.width, camera->viewport.height);
	glScissor(camera->viewport.x, camera->viewport.y, camera->viewport.width, camera->viewport.height);

	glClearColor(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
	glClearDepth(1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->mainFBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->geometryFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->ssaoFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void PhysicsEngine::computeSSAO(World* world, Camera* camera, const std::vector<RenderObject>& renderObjects, ScreenData& screenData, GraphicsQuery& query)
{
	// fill geometry framebuffer
	int shaderProgram = screenData.positionAndNormalsShader.getProgramFromVariant(ShaderVariant::None);
	int modelLoc = screenData.positionAndNormalsShader.findUniformLocation("model", shaderProgram);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->geometryFBO);
	for (size_t i = 0; i < renderObjects.size(); i++) {
		screenData.positionAndNormalsShader.use(shaderProgram);
		screenData.positionAndNormalsShader.setMat4(modelLoc, renderObjects[i].model);

		Graphics::render(world, renderObjects[i], &query);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();

	// fill ssao color texture
	shaderProgram = screenData.ssaoShader.getProgramFromVariant(ShaderVariant::None);
	int projectionLoc = screenData.ssaoShader.findUniformLocation("projection", shaderProgram);
	int positionTexLoc = screenData.ssaoShader.findUniformLocation("positionTex", shaderProgram);
	int normalTexLoc = screenData.ssaoShader.findUniformLocation("normalTex", shaderProgram);
	int noiseTexLoc = screenData.ssaoShader.findUniformLocation("noiseTex", shaderProgram);
	int samplesLoc[64];
	for (int i = 0; i < 64; i++) {
		samplesLoc[i] = screenData.ssaoShader.findUniformLocation("samples[" + std::to_string(i) + "]", shaderProgram);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, camera->ssaoFBO);
	screenData.ssaoShader.use(shaderProgram);
	screenData.ssaoShader.setMat4(projectionLoc, camera->getProjMatrix());
	for (int i = 0; i < 64; i++) {
		screenData.ssaoShader.setVec3(samplesLoc[i], camera->ssaoSamples[i]);
	}
	screenData.ssaoShader.setInt(positionTexLoc, 0);
	screenData.ssaoShader.setInt(normalTexLoc, 1);
	screenData.ssaoShader.setInt(noiseTexLoc, 2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, camera->positionTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, camera->normalTex);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, camera->ssaoNoiseTex);

	glBindVertexArray(screenData.quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::renderShadows(World* world, Camera* camera, Light* light, const std::vector<RenderObject>& renderObjects, ShadowMapData& shadowMapData, GraphicsQuery& query)
{
	LightType lightType = light->lightType;

	if (lightType == LightType::Directional) {

		calcShadowmapCascades(camera, shadowMapData);
		calcCascadeOrthoProj(camera, light, shadowMapData);

		int shaderProgram = shadowMapData.depthShader.getProgramFromVariant(ShaderVariant::None);
		int modelLoc = shadowMapData.depthShader.findUniformLocation("model", shaderProgram);
		int viewLoc = shadowMapData.depthShader.findUniformLocation("view", shaderProgram);
		int projectionLoc = shadowMapData.depthShader.findUniformLocation("projection", shaderProgram);

		for (int i = 0; i < 5; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowCascadeFBO[i]);

			glClearDepth(1.0f);
			glClear(GL_DEPTH_BUFFER_BIT);

			shadowMapData.depthShader.use(shaderProgram);
			shadowMapData.depthShader.setMat4(viewLoc, shadowMapData.cascadeLightView[i]);
			shadowMapData.depthShader.setMat4(projectionLoc, shadowMapData.cascadeOrthoProj[i]);

			for (int j = 0; j < renderObjects.size(); j++) {
				shadowMapData.depthShader.setMat4(modelLoc, renderObjects[j].model);
				Graphics::render(world, renderObjects[j], &query);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	}
	else if (lightType == LightType::Spot) {

		int shaderProgram = shadowMapData.depthShader.getProgramFromVariant(ShaderVariant::None);
		int modelLoc = shadowMapData.depthShader.findUniformLocation("model", shaderProgram);
		int viewLoc = shadowMapData.depthShader.findUniformLocation("view", shaderProgram);
		int projectionLoc = shadowMapData.depthShader.findUniformLocation("projection", shaderProgram);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowSpotlightFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadowMapData.shadowProjMatrix = light->getProjMatrix();
		shadowMapData.shadowViewMatrix = glm::lookAt(light->position, light->position + light->direction, glm::vec3(0.0f, 1.0f, 0.0f));

		shadowMapData.depthShader.use(shaderProgram);
		shadowMapData.depthShader.setMat4(projectionLoc, shadowMapData.shadowProjMatrix);
		shadowMapData.depthShader.setMat4(viewLoc, shadowMapData.shadowViewMatrix);

		for (int i = 0; i < renderObjects.size(); i++) {
			shadowMapData.depthShader.setMat4(modelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	else if (lightType == LightType::Point) {

		shadowMapData.cubeViewProjMatrices[0] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.cubeViewProjMatrices[1] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.cubeViewProjMatrices[2] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
		shadowMapData.cubeViewProjMatrices[3] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
		shadowMapData.cubeViewProjMatrices[4] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
		shadowMapData.cubeViewProjMatrices[5] = (light->getProjMatrix() * glm::lookAt(light->position, light->position + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));

		int shaderProgram = shadowMapData.depthCubemapShader.getProgramFromVariant(ShaderVariant::None);
		int lightPosLoc = shadowMapData.depthCubemapShader.findUniformLocation("lightPos", shaderProgram);
		int farPlaneLoc = shadowMapData.depthCubemapShader.findUniformLocation("farPlane", shaderProgram);
		int modelLoc = shadowMapData.depthCubemapShader.findUniformLocation("model", shaderProgram);
		int cubeViewProjMatricesLoc0 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[0]", shaderProgram);
		int cubeViewProjMatricesLoc1 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[1]", shaderProgram);
		int cubeViewProjMatricesLoc2 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[2]", shaderProgram);
		int cubeViewProjMatricesLoc3 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[3]", shaderProgram);
		int cubeViewProjMatricesLoc4 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[4]", shaderProgram);
		int cubeViewProjMatricesLoc5 = shadowMapData.depthCubemapShader.findUniformLocation("cubeViewProjMatrices[5]", shaderProgram);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapData.shadowCubemapFBO);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadowMapData.depthCubemapShader.use(shaderProgram);
		shadowMapData.depthCubemapShader.setVec3(lightPosLoc, light->position);
		shadowMapData.depthCubemapShader.setFloat(farPlaneLoc, camera->frustum.farPlane);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc0, shadowMapData.cubeViewProjMatrices[0]);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc1, shadowMapData.cubeViewProjMatrices[1]);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc2, shadowMapData.cubeViewProjMatrices[2]);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc3, shadowMapData.cubeViewProjMatrices[3]);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc4, shadowMapData.cubeViewProjMatrices[4]);
		shadowMapData.depthCubemapShader.setMat4(cubeViewProjMatricesLoc5, shadowMapData.cubeViewProjMatrices[5]);

		for (int i = 0; i < renderObjects.size(); i++) {
			shadowMapData.depthCubemapShader.setMat4(modelLoc, renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void PhysicsEngine::renderOpaques(World* world, Camera* camera, Light* light, const std::vector<RenderObject>& renderObjects, const ShadowMapData& shadowMapData, GraphicsLightState& lightState, GraphicsQuery& query)
{
	lightState.position = light->position;
	lightState.direction = light->direction;
	lightState.ambient = light->ambient;
	lightState.diffuse = light->diffuse;
	lightState.specular = light->specular;

	lightState.constant = light->constant;
	lightState.linear = light->linear;
	lightState.quadratic = light->quadratic;
	lightState.cutOff = light->cutOff;
	lightState.outerCutOff = light->outerCutOff;

	if (light->lightType == LightType::Directional) {
		for (int i = 0; i < 5; i++) {
			lightState.lightProjection[i] = shadowMapData.cascadeOrthoProj[i];

			glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, shadowMapData.cascadeEnds[i + 1], 1.0f);
			glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
			lightState.cascadeEnds[i] = clipSpaceCascadeEnd.z;

			lightState.lightView[i] = shadowMapData.cascadeLightView[i];
		}
	}
	else if (light->lightType == LightType::Spot) {
		for (int i = 0; i < 5; i++) {
			lightState.lightProjection[i] = shadowMapData.shadowProjMatrix;
			lightState.lightView[i] = shadowMapData.shadowViewMatrix;
		}
	}

	lightState.farPlane = camera->frustum.farPlane;

	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &lightState.lightProjection[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &lightState.lightView[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(lightState.position));
	glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(lightState.direction));
	glBufferSubData(GL_UNIFORM_BUFFER, 672, 12, glm::value_ptr(lightState.ambient));
	glBufferSubData(GL_UNIFORM_BUFFER, 688, 12, glm::value_ptr(lightState.diffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 704, 12, glm::value_ptr(lightState.specular));
	glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &lightState.cascadeEnds[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &lightState.cascadeEnds[1]);
	glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &lightState.cascadeEnds[2]);
	glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &lightState.cascadeEnds[3]);
	glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &lightState.cascadeEnds[4]);
	glBufferSubData(GL_UNIFORM_BUFFER, 800, 4, &lightState.farPlane);
	glBufferSubData(GL_UNIFORM_BUFFER, 804, 4, &(lightState.constant));
	glBufferSubData(GL_UNIFORM_BUFFER, 808, 4, &(lightState.linear));
	glBufferSubData(GL_UNIFORM_BUFFER, 812, 4, &(lightState.quadratic));
	glBufferSubData(GL_UNIFORM_BUFFER, 816, 4, &(lightState.cutOff));
	glBufferSubData(GL_UNIFORM_BUFFER, 820, 4, &(lightState.outerCutOff));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	int variant = ShaderVariant::None;
	if (light->lightType == LightType::Directional) {
		variant = ShaderVariant::Directional;
	}
	else if (light->lightType == LightType::Spot) {
		variant = ShaderVariant::Spot;
	}
	else if(light->lightType == LightType::Point) {
		variant = ShaderVariant::Point;
	}

	if (light->shadowType == ShadowType::Hard) {
		variant |= ShaderVariant::HardShadows;
	}
	else if (light->shadowType == ShadowType::Soft) {
		variant |= ShaderVariant::SoftShadows;
	}

	const std::string shaderShadowMapNames[] = { "shadowMap[0]",
												 "shadowMap[1]",
												 "shadowMap[2]",
												 "shadowMap[3]",
												 "shadowMap[4]" };

	glBindFramebuffer(GL_FRAMEBUFFER, camera->mainFBO);

	for (int i = 0; i < renderObjects.size(); i++) {
		Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);
		Shader* shader = world->getAssetByIndex<Shader>(renderObjects[i].shaderIndex);

		int shaderProgram = shader->getProgramFromVariant(variant);

		shader->use(shaderProgram);
		shader->setMat4("model", renderObjects[i].model);
		
		material->apply(world);
		//material->use(shader, renderObjects[i]);

		if (light->lightType == LightType::Directional) {
			for (int j = 0; j < 5; j++) {
				shader->setInt(shaderShadowMapNames[j], 3 + j);

				glActiveTexture(GL_TEXTURE0 + 3 + j);
				glBindTexture(GL_TEXTURE_2D, shadowMapData.shadowCascadeDepth[j]);
			}
		}
		else if (light->lightType == LightType::Spot) {
			shader->setInt(shaderShadowMapNames[0], 3);

			glActiveTexture(GL_TEXTURE0 + 3);
			glBindTexture(GL_TEXTURE_2D, shadowMapData.shadowSpotlightDepth);
		}

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

void PhysicsEngine::endFrame(World* world, Camera* camera, const std::vector<RenderObject>& renderObjects, ScreenData& screenData, GraphicsTargets& targets, GraphicsDebug& debug, GraphicsQuery& query, bool renderToScreen)
{

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
	targets.color = camera->colorTex;
	targets.depth = camera->depthTex;
	targets.position = camera->positionTex;
	targets.normals = camera->normalTex;
	targets.overdraw = -1;
	targets.ssao = camera->ssaoColorTex;
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

		screenData.quadShader.use(ShaderVariant::None);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, camera->colorTex);

		glBindVertexArray(screenData.quadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}

void PhysicsEngine::calcShadowmapCascades(Camera* camera, ShadowMapData& shadowMapData)
{
	float nearDist = camera->frustum.nearPlane;
	float farDist = camera->frustum.farPlane;

	const float splitWeight = 0.95f;
	const float ratio = farDist / nearDist;

	for (int i = 0; i < 6; i++) {
		const float si = i / 5.0f;

		shadowMapData.cascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) + (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
	}
}

void PhysicsEngine::calcCascadeOrthoProj(Camera* camera, Light* light, ShadowMapData& shadowMapData)
{
	glm::mat4 viewInv = glm::inverse(camera->getViewMatrix());
	float fov = camera->frustum.fov;
	float aspect = camera->viewport.getAspectRatio();
	float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
	float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

	glm::vec3 direction = light->direction;

	for (unsigned int i = 0; i < 5; i++) {
		float xn = -1.0f * shadowMapData.cascadeEnds[i] * tanHalfHFOV;
		float xf = -1.0f * shadowMapData.cascadeEnds[i + 1] * tanHalfHFOV;
		float yn = -1.0f * shadowMapData.cascadeEnds[i] * tanHalfVFOV;
		float yf = -1.0f * shadowMapData.cascadeEnds[i + 1] * tanHalfVFOV;

		glm::vec4 frustumCorners[8];
		frustumCorners[0] = glm::vec4(xn, yn, shadowMapData.cascadeEnds[i], 1.0f);
		frustumCorners[1] = glm::vec4(-xn, yn, shadowMapData.cascadeEnds[i], 1.0f);
		frustumCorners[2] = glm::vec4(xn, -yn, shadowMapData.cascadeEnds[i], 1.0f);
		frustumCorners[3] = glm::vec4(-xn, -yn, shadowMapData.cascadeEnds[i], 1.0f);

		frustumCorners[4] = glm::vec4(xf, yf, shadowMapData.cascadeEnds[i + 1], 1.0f);
		frustumCorners[5] = glm::vec4(-xf, yf, shadowMapData.cascadeEnds[i + 1], 1.0f);
		frustumCorners[6] = glm::vec4(xf, -yf, shadowMapData.cascadeEnds[i + 1], 1.0f);
		frustumCorners[7] = glm::vec4(-xf, -yf, shadowMapData.cascadeEnds[i + 1], 1.0f);

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

		glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d * direction.x, frustrumCentreWorldSpace.y + d * direction.y, frustrumCentreWorldSpace.z + d * direction.z);

		shadowMapData.cascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace.x, frustrumCentreWorldSpace.y, frustrumCentreWorldSpace.z), glm::vec3(1.0f, 0.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();
		for (unsigned int j = 0; j < 8; j++) {
			// Transform the frustum coordinate from view to world space
			glm::vec4 vW = viewInv * frustumCorners[j];

			//std::cout << "j: " << j << " " << vW.x << " " << vW.y << " " << vW.z << " " << vW.w << std::endl;

			// Transform the frustum coordinate from world to light space
			glm::vec4 vL = shadowMapData.cascadeLightView[i] * vW;

			//std::cout << "j: " << j << " " << vL.x << " " << vL.y << " " << vL.z << " " << vL.w << std::endl;

			minX = glm::min(minX, vL.x);
			maxX = glm::max(maxX, vL.x);
			minY = glm::min(minY, vL.y);
			maxY = glm::max(maxY, vL.y);
			minZ = glm::min(minZ, vL.z);
			maxZ = glm::max(maxZ, vL.z);
		}

		// std::cout << "i: " << i << " " << minX << " " << maxX << " " << minY << " " << maxY << " " << minZ << " " << maxZ << "      " << p.x << " " << p.y << " " << p.z << "      " << frustrumCentreWorldSpace.x << " " << frustrumCentreWorldSpace.y << " " << frustrumCentreWorldSpace.z << std::endl;

		shadowMapData.cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
	}
}


void PhysicsEngine::addToRenderObjectsList(World* world, MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects)
{
	Transform* transform = meshRenderer->getComponent<Transform>(world);
	if (transform == NULL) {
		std::string message = "Error: Could not find transform for meshrenderer with id " + meshRenderer->getId().toString() + "\n";
		return;
	}

	Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);
	if (mesh == NULL) {
		std::string message = "Error: Could not find mesh with id " + meshRenderer->meshId.toString() + "\n";
		return;
	}

	int transformIndex = world->getIndexOf(transform->getId());
	int meshStartIndex = 0;
	Sphere test;
	Sphere boundingSphere = test;// mesh->getBoundingSphere();

	for (int i = 0; i < meshRenderer->materialCount; i++) {
		int materialIndex = world->getIndexOf(meshRenderer->materialIds[i]);
		int subMeshVertexStartIndex = mesh->getSubMeshStartIndices()[i];
		int subMeshVertexEndIndex = mesh->getSubMeshStartIndices()[i + 1];
		int subMeshVerticesCount = subMeshVertexEndIndex - subMeshVertexStartIndex;

		Material* material = world->getAssetByIndex<Material>(materialIndex);
		Shader* shader = world->getAsset<Shader>(material->getShaderId());

		int shaderIndex = world->getIndexOf(shader->getId());

		RenderObject renderObject;
		renderObject.id = meshRenderer->getId();
		renderObject.start = meshStartIndex + subMeshVertexStartIndex;
		renderObject.size = subMeshVerticesCount;
		renderObject.transformIndex = transformIndex;
		renderObject.materialIndex = materialIndex;
		renderObject.shaderIndex = shaderIndex;
		renderObject.vao = mesh->vao;

		renderObject.mainTexture = -1;
		renderObject.normalMap = -1;
		renderObject.specularMap = -1;

		/*Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
		Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
		Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);

		if (mainTexture != NULL) { renderObject.mainTexture = mainTexture->handle.handle; }
		if (normalMap != NULL) { renderObject.normalMap = normalMap->handle.handle; }
		if (specularMap != NULL) { renderObject.specularMap = specularMap->handle.handle; }*/

		renderObject.boundingSphere = boundingSphere;

		renderObjects.push_back(renderObject);
	}
}

void PhysicsEngine::removeFromRenderObjectsList(MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects)
{
	//mmm this is slow...need a faster way of removing render objects
	for (size_t i = 0; i < renderObjects.size(); i++) {
		if (meshRenderer->getId() == renderObjects[i].id) {
			renderObjects.erase(renderObjects.begin() + i);
		}
	}
}