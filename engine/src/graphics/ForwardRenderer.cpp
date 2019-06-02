#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Line.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{

}

ForwardRenderer::~ForwardRenderer()
{

}

void ForwardRenderer::init(World* world)
{
	this->world = world;

	// grab camera
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	glGenQueries(1, &(query.queryId));

	// generate all texture assets
	createTextures();

	// compile all shader assets and configure uniform blocks
	createShaderPrograms();

	// generate one large vertex buffer storing all unique mesh vertices, normals, and tex coordinates
	createMeshBuffers();

	// generate fbo
	createMainFBO();

    // generate shadow map fbos
    createShadowMapFBOs();

	// batch all static meshes by material
	// for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
	// 	MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

	// 	if(meshRenderer != NULL && meshRenderer->isStatic){
	// 		Transform* transform = meshRenderer->getComponent<Transform>(world);
	// 		Material* material = world->getAsset<Material>(meshRenderer->materialId);
	// 		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

	// 		glm::mat4 model = transform->getModelMatrix();

	// 		batchManager.add(material, mesh, model);
	// 	}
	// }

	// generate render objects list
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		if(meshRenderer != NULL){
			add(meshRenderer);
		}
	}

	initCameraUniformState();
	initLightUniformState();

	debug.init();

	Graphics::checkError();
}

void ForwardRenderer::update(Input input)
{
	if(camera == NULL) { return; }

	std::cout << "camera position: " << camera->position.x << " " << camera->position.y << " " << camera->position.z << std::endl;

	query.numBatchDrawCalls = 0;
	query.numDrawCalls = 0;
	query.totalElapsedTime = 0.0f;
	query.verts = 0;
	query.tris = 0;
	query.lines = 0;
	query.points = 0;

	// cull render objects based on camera frustrum
	for(size_t i = 0; i < renderObjects.size(); i++){
		Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();

		glm::vec4 temp = renderObjects[i].model * glm::vec4(renderObjects[i].boundingSphere.centre, 1.0f);
		glm::vec3 centre = glm::vec3(temp.x, temp.y, temp.z);
		float radius = renderObjects[i].boundingSphere.radius;

		//if(camera->checkSphereInFrustum(centre, radius)){
		//	std::cout << "Render object inside camera frustrum " << centre.x << " " << centre.y << " " << centre.z << " " << radius << std::endl;
		//}
	}

	beginFrame(camera, fbo);

	pass = 0;

	renderDirectionalLights();
	renderSpotLights();
	renderPointLights();

	if(getKeyDown(input, KeyCode::Z)){
		std::cout << "recording depth texture data" << std::endl;
		std::vector<unsigned char> data;
		data.resize(1024*1024);
		glGetTextureImage(shadowCascadeDepth[0], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		World::writeToBMP("shadow_depth_data0.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[1], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		World::writeToBMP("shadow_depth_data1.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[2], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		World::writeToBMP("shadow_depth_data2.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[3], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		World::writeToBMP("shadow_depth_data3.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[4], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		World::writeToBMP("shadow_depth_data4.bmp", data, 1024, 1024, 1);
	}

	if(world->debug){
		renderDebug(world->debugView);
	}

	endFrame(color);

    Graphics::checkError();
}

void ForwardRenderer::add(MeshRenderer* meshRenderer)
{
	if(meshRenderer != NULL && !meshRenderer->isStatic){
		Transform* transform = meshRenderer->getComponent<Transform>(world);

		int transformIndex = world->getIndexOf(transform->componentId);
		int materialIndex = world->getIndexOfAsset(meshRenderer->materialId);

		Material* material = world->getAssetByIndex<Material>(materialIndex);

		if(material == NULL){
			std::cout << "Error: Trying to add mesh renderer with null material" << std::endl;
			return;
		}
		
		Shader* shader = world->getAsset<Shader>(material->shaderId);

		if(shader == NULL){
			std::cout << "Error: Trying to add mesh renderer with null shader" << std::endl;
			return;
		}

		Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
		Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
		Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);

		int index = meshBuffer.getIndex(meshRenderer->meshId);

		std::cout << "transform index: " << transformIndex << " " << materialIndex << " mesh id: " << meshRenderer->meshId.toString() << " index: " << index << std::endl;

		std::cout << "index: " << index << "  " << meshBuffer.start[index] << " " << meshBuffer.count[index] << "  " << transformIndex << " " << materialIndex << std::endl;

		if(index != -1){
			RenderObject renderObject;
			renderObject.start = meshBuffer.start[index];
			renderObject.size = meshBuffer.count[index];
			renderObject.transformIndex = transformIndex;
			renderObject.materialIndex = materialIndex;

			for(int i = 0; i < 4; i++){
				renderObject.shaders[i] = shader->programs[i].handle;
			}

			renderObject.mainTexture = -1;
			renderObject.normalMap = -1;
			renderObject.specularMap = -1;
			
			if(mainTexture != NULL){ renderObject.mainTexture = mainTexture->handle.handle; }
			if(normalMap != NULL){ renderObject.normalMap = normalMap->handle.handle; }
			if(specularMap != NULL){ renderObject.specularMap = specularMap->handle.handle; }

			renderObject.boundingSphere = meshBuffer.boundingSpheres[index];

			renderObjects.push_back(renderObject);
		}		
	}
}

void ForwardRenderer::remove(MeshRenderer* meshRenderer)
{

}

GraphicsQuery ForwardRenderer::getGraphicsQuery()
{
	return query;
}

GraphicsDebug ForwardRenderer::getGraphicsDebug()
{
	return debug;
}

void ForwardRenderer::beginFrame(Camera* camera, GLuint fbo)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(camera->viewport.x, camera->viewport.y, camera->viewport.width, camera->viewport.height);
	glClearColor(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	updateCameraUniformState(camera);
}

void ForwardRenderer::endFrame(GLuint tex)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Graphics::use(&quadShader, ShaderVariant::None);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void ForwardRenderer::renderDirectionalLights()
{
	if (world->getNumberOfComponents<DirectionalLight>() > 0){
		DirectionalLight* directionalLight = world->getComponentByIndex<DirectionalLight>(0);

		//lightView = glm::lookAt(directionalLight->direction, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		// lightView = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), directionalLight->direction, glm::vec3(0.0f, 1.0f, 0.0f));
		// calcCascadeOrthoProj(camera->getViewMatrix(), lightView, directionalLight->direction);
		calcShadowmapCascades(camera->frustum.nearPlane, camera->frustum.farPlane);
		calcCascadeOrthoProj(camera->getViewMatrix(), directionalLight->direction);

		for(int j = 0; j < 5; j++){
			renderShadowMap(shadowCascadeFBO[j], cascadeLightView[j], cascadeOrthoProj[j]);
		}

		updateLightUniformState(directionalLight);

		ShaderVariant variant = ShaderVariant::Directional;
		if(directionalLight->shadowType == ShadowType::Hard){
			variant = ShaderVariant::Directional_Hard;
		}
		else if(directionalLight->shadowType == ShadowType::Soft){
			variant = ShaderVariant::Directional_Soft;
		}

		render(fbo, variant);

		pass++;
	}
}

void ForwardRenderer::renderSpotLights()
{
	for(int i = 0; i < world->getNumberOfComponents<SpotLight>(); i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		SpotLight* spotLight = world->getComponentByIndex<SpotLight>(i);

		updateLightUniformState(spotLight);

		ShaderVariant variant = ShaderVariant::Spot;
		if(spotLight->shadowType == ShadowType::Hard){
			variant = ShaderVariant::Spot_Hard;
		}
		else if(spotLight->shadowType == ShadowType::Soft){
			variant = ShaderVariant::Spot_Soft;
		}

		render(fbo, variant);

		pass++;
	}
}

void ForwardRenderer::renderPointLights()
{
	for(int i = 0; i < world->getNumberOfComponents<PointLight>(); i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		PointLight* pointLight = world->getComponentByIndex<PointLight>(i);

		updateLightUniformState(pointLight);

		ShaderVariant variant = ShaderVariant::Point;
		if(pointLight->shadowType == ShadowType::Hard){
			variant = ShaderVariant::Point_Hard;
		}
		else if(pointLight->shadowType == ShadowType::Soft){
			variant = ShaderVariant::Point_Soft;
		}

		render(fbo, variant);
			
		pass++;
	}
}

void ForwardRenderer::render(GLuint fbo, ShaderVariant variant)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glBindVertexArray(meshBuffer.vao);

	for(int i = 0; i < renderObjects.size(); i++){
		// Graphics::render(world, renderObjects[i], variant, &query);
		Graphics::render(world, renderObjects[i], variant, &shadowCascadeDepth[0], 5, &query);
	}

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ForwardRenderer::renderShadowMap(GLuint fbo, glm::mat4 lightView, glm::mat4 lightProjection)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glEnable(GL_DEPTH_TEST); 
	//glDepthFunc(GL_LEQUAL);
	//glDepthFunc(GL_ALWAYS);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(meshBuffer.vao);

	for(int i = 0; i < renderObjects.size(); i++){
		Graphics::render(world, &depthShader, ShaderVariant::None, renderObjects[i].model, lightView, lightProjection, renderObjects[i].start, renderObjects[i].size, &query);
	}

	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ForwardRenderer::renderDebug(int view)
{		
	glBindFramebuffer(GL_FRAMEBUFFER, debug.fbo[view].handle);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glClearDepth(1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	if(view == 0 || view == 1 || view == 2){
		batchManager.render(world, &debug.shaders[view], ShaderVariant::None, NULL);

		//render non static meshes here
		glBindVertexArray(meshBuffer.vao);

		for(int i = 0; i < renderObjects.size(); i++){
			Graphics::render(world, &debug.shaders[view], ShaderVariant::None, renderObjects[i].model, renderObjects[i].start, renderObjects[i].size, &query);
		}

		glBindVertexArray(0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ForwardRenderer::createTextures()
{
	for(int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++){
		Texture2D* texture = world->getAssetByIndex<Texture2D>(i);
		if(texture != NULL){
			int width = texture->getWidth();
			int height = texture->getHeight();
			int numChannels = texture->getNumChannels();
			TextureFormat format = texture->getFormat();
			std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

			glGenTextures(1, &(texture->handle.handle));
			glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

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
				std::cout << "Error: Invalid texture format" << std::endl;
			}

			glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

			glGenerateMipmap(GL_TEXTURE_2D);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			glBindTexture(GL_TEXTURE_2D, 0);
		}
	}
}

void ForwardRenderer::createShaderPrograms()
{
	for(int i = 0; i < world->getNumberOfAssets<Shader>(); i++){
		Shader* shader = world->getAssetByIndex<Shader>(i);

		if(shader != NULL){
			shader->compile();

			if(!shader->isCompiled()){
				std::cout << "Shader failed to compile " << i << " " << shader->assetId.toString() << std::endl;
			}

			std::string uniformBlocks[] = {"CameraBlock", 
										   "LightBlock"};

			for(int i = 0; i < 2; i++){
				shader->setUniformBlock(uniformBlocks[i], i);
			}
		}
	}

	depthShader.vertexShader = Shader::shadowDepthMapVertexShader;
	depthShader.fragmentShader = Shader::shadowDepthMapFragmentShader;
	depthShader.compile();

	quadShader.vertexShader = Shader::windowVertexShader;
	quadShader.fragmentShader = Shader::windowFragmentShader;
	quadShader.compile();
}

void ForwardRenderer::createMeshBuffers()
{
	size_t totalVerticesSize = 0;
	size_t totalNormalsSize = 0;
	size_t totalTexCoordsSize = 0;
	for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		totalVerticesSize += mesh->vertices.size();
		totalNormalsSize += mesh->normals.size();
		totalTexCoordsSize += mesh->texCoords.size();
	}

	meshBuffer.vertices.reserve(totalVerticesSize);
	meshBuffer.normals.reserve(totalNormalsSize);
	meshBuffer.texCoords.reserve(totalTexCoordsSize);

	int startIndex = 0;
	for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		std::cout << "adding mesh: " << mesh->assetId.toString() << " start index: " << startIndex << std::endl;

		meshBuffer.meshIds.push_back(mesh->assetId);
		meshBuffer.start.push_back(startIndex);
		meshBuffer.count.push_back((int)mesh->vertices.size());
		meshBuffer.vertices.insert(meshBuffer.vertices.end(), mesh->vertices.begin(), mesh->vertices.end());
		meshBuffer.normals.insert(meshBuffer.normals.end(), mesh->normals.begin(), mesh->normals.end());
		meshBuffer.texCoords.insert(meshBuffer.texCoords.end(), mesh->texCoords.begin(), mesh->texCoords.end());
		meshBuffer.boundingSpheres.push_back(mesh->getBoundingSphere());

		startIndex += (int)mesh->vertices.size();
	}

	meshBuffer.init();

	std::cout << "mesh buffer size: " << meshBuffer.vertices.size() << " " << meshBuffer.normals.size() << " " << meshBuffer.texCoords.size() << std::endl;
}

void ForwardRenderer::createMainFBO()
{
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glGenTextures(1, &color);
	glBindTexture(GL_TEXTURE_2D, color);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera->viewport.width, camera->viewport.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, camera->viewport.width, camera->viewport.height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[1] = { GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, attachments);

	Graphics::checkFrambufferError();
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//generate screen quad for final rendering
	float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };

	glGenVertexArrays(1, &quadVAO);
	glBindVertexArray(quadVAO);

	std::cout << "quadVAO: " << quadVAO << std::endl;

	glGenBuffers(1, &quadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void ForwardRenderer::createShadowMapFBOs()
{
	// create directional light cascade shadow map fbo
	glGenFramebuffers(5, &shadowCascadeFBO[0]);
	glGenTextures(5, &shadowCascadeDepth[0]);

	for(int i = 0; i < 5; i++){
		glBindFramebuffer(GL_FRAMEBUFFER, shadowCascadeFBO[i]);
		glBindTexture(GL_TEXTURE_2D, shadowCascadeDepth[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowCascadeDepth[i], 0);

		Graphics::checkFrambufferError();
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	std::cout << "Cascade shadow maps created" << std::endl;

	// create spotlight shadow map fbo
	glGenFramebuffers(1, &shadowSpotlightFBO);
	glGenTextures(1, &shadowSpotlightDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowSpotlightFBO);
	glBindTexture(GL_TEXTURE_2D, shadowSpotlightDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowSpotlightDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::cout << "Spotlight shadow maps created" << std::endl;

	// create pointlight shadow cubemap fbo
	glGenFramebuffers(1, &shadowCubemapFBO);
	glGenTextures(1, &shadowCubemapDepth);

	glBindFramebuffer(GL_FRAMEBUFFER, shadowCubemapFBO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowCubemapDepth);
	for (unsigned int i = 0; i < 6; i++){
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	}

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowCubemapDepth, 0);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::cout << "Pointlight shadow maps created" << std::endl;
}

void ForwardRenderer::calcShadowmapCascades(float nearDist, float farDist)
{
	const float splitWeight = 0.95f;
    const float ratio = farDist / nearDist;

    for(int i = 0; i < 6; i++){
    	const float si = i / 5.0f;

    	cascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) + (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));

    	std::cout << "i: " << i << " " << cascadeEnds[i] << std::endl;
    }
}

void ForwardRenderer::calcCascadeOrthoProj(glm::mat4 view, glm::vec3 direction)
{
	// std::cout << view[0][0] << " " << view[0][1] << " " << view[0][2] << " " << view[0][3] << std::endl;
	// std::cout << view[1][0] << " " << view[1][1] << " " << view[1][2] << " " << view[1][3] << std::endl;
	// std::cout << view[2][0] << " " << view[2][1] << " " << view[2][2] << " " << view[2][3] << std::endl;
	// std::cout << view[3][0] << " " << view[3][1] << " " << view[3][2] << " " << view[3][3] << std::endl;

	glm::mat4 viewInv = glm::inverse(view);
	float fov = camera->frustum.fov;
	float aspect = camera->frustum.aspectRatio;
	float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
	float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

	//std::cout << "fov: " << fov << " aspect: " << aspect << " tanHalfHFOV: " << tanHalfHFOV << " tanHalfVFOV: " << tanHalfVFOV << std::endl;
	// float factor = glm::tan(glm::radians(45.0f));

	for (unsigned int i = 0; i < 5; i++){
		float xn = -1.0f * cascadeEnds[i] * tanHalfHFOV;
		float xf = -1.0f * cascadeEnds[i + 1] * tanHalfHFOV;
		float yn = -1.0f * cascadeEnds[i] * tanHalfVFOV;
		float yf = -1.0f * cascadeEnds[i + 1] * tanHalfVFOV;

		//std::cout << "i: " << i << " xn: " << xn << " xf: " << xf << " yn: " << yn << " yf: " << yf << std::endl;

		glm::vec4 frustumCorners[8];
		frustumCorners[0] = glm::vec4(xn, yn, cascadeEnds[i], 1.0f);
		frustumCorners[1] = glm::vec4(-xn, yn, cascadeEnds[i], 1.0f);
		frustumCorners[2] = glm::vec4(xn, -yn, cascadeEnds[i], 1.0f);
		frustumCorners[3] = glm::vec4(-xn, -yn, cascadeEnds[i], 1.0f);

		frustumCorners[4] = glm::vec4(xf, yf, cascadeEnds[i + 1], 1.0f);
		frustumCorners[5] = glm::vec4(-xf, yf, cascadeEnds[i + 1], 1.0f);
		frustumCorners[6] = glm::vec4(xf, -yf, cascadeEnds[i + 1], 1.0f);
		frustumCorners[7] = glm::vec4(-xf, -yf, cascadeEnds[i + 1], 1.0f);

		glm::vec4 frustumCentre = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		for (int j = 0; j < 8; j++){
			frustumCentre.x += frustumCorners[j].x;
			frustumCentre.y += frustumCorners[j].y;
			frustumCentre.z += frustumCorners[j].z;
		}

		frustumCentre.x = frustumCentre.x / 8;
		frustumCentre.y = frustumCentre.y / 8;
		frustumCentre.z = frustumCentre.z / 8;

		// Transform the frustum centre from view to world space
		glm::vec4 frustrumCentreWorldSpace = viewInv * frustumCentre;
		float d = 20.0f;//cascadeEnds[i + 1] - cascadeEnds[i];

		glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d*direction.x, frustrumCentreWorldSpace.y + d*direction.y, frustrumCentreWorldSpace.z + d*direction.z);

		cascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace.x, frustrumCentreWorldSpace.y, frustrumCentreWorldSpace.z), glm::vec3(0.0f, 1.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();
		for (unsigned int j = 0; j < 8; j++){
			// Transform the frustum coordinate from view to world space
			glm::vec4 vW = viewInv * frustumCorners[j];

			// if(i == 0){
			// 	std::cout << "j: " << j << " " << vW.x << " " << vW.y << " " << vW.z << " " << vW.w << std::endl;
			// }
			//std::cout << "j: " << j << " " << vW.x << " " << vW.y << " " << vW.z << " " << vW.w << std::endl;

			// Transform the frustum coordinate from world to light space
			// glm::vec4 vL = lightView * vW;
			glm::vec4 vL = cascadeLightView[i] * vW;

			//std::cout << "j: " << j << " " << vL.x << " " << vL.y << " " << vL.z << " " << vL.w << std::endl;

			minX = glm::min(minX, vL.x);
			maxX = glm::max(maxX, vL.x);
			minY = glm::min(minY, vL.y);
			maxY = glm::max(maxY, vL.y);
			minZ = glm::min(minZ, vL.z);
			maxZ = glm::max(maxZ, vL.z);
		}

		//std::cout << "i: " << i << " " << minX << " " << maxX << " " << minY << " " << maxY << " " << minZ << " " << maxZ << "      " << p.x << " " << p.y << " " << p.z << "      " << frustrumCentreWorldSpace.x << " " << frustrumCentreWorldSpace.y << " " << frustrumCentreWorldSpace.z << std::endl;

		cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
		// cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, -maxZ, -minZ);
		//cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, minZ, maxZ);
	}
}

void ForwardRenderer::calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection)
{
	cubeViewMatrices[0] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[1] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[2] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
	cubeViewMatrices[3] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
	cubeViewMatrices[4] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[5] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
}

void ForwardRenderer::initCameraUniformState()
{
	glGenBuffers(1, &(cameraState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraState.handle, 0, 144);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::initLightUniformState()
{
	glGenBuffers(1, &(lightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, lightState.handle, 0, 824);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateCameraUniformState(Camera* camera)
{
	cameraState.projection = camera->getProjMatrix();
	cameraState.view = camera->getViewMatrix();
	cameraState.cameraPos = camera->position;

	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(cameraState.projection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(cameraState.view));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(cameraState.cameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateLightUniformState(DirectionalLight* light)
{
	lightState.direction = light->direction;
	lightState.ambient = light->ambient;
	lightState.diffuse = light->diffuse;
	lightState.specular = light->specular;

	for(int i = 0; i < 5; i++){
		lightState.lightProjection[i] = cascadeOrthoProj[i];

		glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, cascadeEnds[i+1], 1.0f);
		glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
		lightState.cascadeEnds[i] = clipSpaceCascadeEnd.z;

		lightState.lightView[i] = cascadeLightView[i];
	}

	lightState.farPlane = camera->frustum.farPlane;

	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &lightState.lightProjection[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &lightState.lightView[0]);
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
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateLightUniformState(SpotLight* light)
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

	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(lightState.position));
	glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(lightState.direction));
	glBufferSubData(GL_UNIFORM_BUFFER, 672, 12, glm::value_ptr(lightState.ambient));
	glBufferSubData(GL_UNIFORM_BUFFER, 688, 12, glm::value_ptr(lightState.diffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 704, 12, glm::value_ptr(lightState.specular));
	glBufferSubData(GL_UNIFORM_BUFFER, 804, 4, &(lightState.constant));
	glBufferSubData(GL_UNIFORM_BUFFER, 808, 4, &(lightState.linear));
	glBufferSubData(GL_UNIFORM_BUFFER, 812, 4, &(lightState.quadratic));
	glBufferSubData(GL_UNIFORM_BUFFER, 816, 4, &(lightState.cutOff));
	glBufferSubData(GL_UNIFORM_BUFFER, 820, 4, &(lightState.outerCutOff));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}	

void ForwardRenderer::updateLightUniformState(PointLight* light)
{
	lightState.position = light->position;
	lightState.ambient = light->ambient;
	lightState.diffuse = light->diffuse;
	lightState.specular = light->specular;
	lightState.constant = light->constant;
	lightState.linear = light->linear;
	lightState.quadratic = light->quadratic;

	glBindBuffer(GL_UNIFORM_BUFFER, lightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(lightState.position));
	glBufferSubData(GL_UNIFORM_BUFFER, 672, 12, glm::value_ptr(lightState.ambient));
	glBufferSubData(GL_UNIFORM_BUFFER, 688, 12, glm::value_ptr(lightState.diffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 704, 12, glm::value_ptr(lightState.specular));
	glBufferSubData(GL_UNIFORM_BUFFER, 804, 4, &lightState.constant);
	glBufferSubData(GL_UNIFORM_BUFFER, 808, 4, &lightState.linear);
	glBufferSubData(GL_UNIFORM_BUFFER, 812, 4, &lightState.quadratic);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}