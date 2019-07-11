#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Line.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Util.h"

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

	// generate ssao fbo
	createSSAOFBO();

    // generate shadow map fbos
    createShadowMapFBOs();

    initRenderObjectsList();

	initCameraUniformState();
	initLightUniformState();

	debug.init();

	Graphics::checkError();
}

void ForwardRenderer::update(Input input)
{
	if(camera == NULL) { return; }

	query.numBatchDrawCalls = 0;
	query.numDrawCalls = 0;
	query.totalElapsedTime = 0.0f;
	query.verts = 0;
	query.tris = 0;
	query.lines = 0;
	query.points = 0;

	updateRenderObjectsList();

	cullingPass();

	beginFrame(camera, fbo);

	pass = 0;

	for(int i = 0; i < world->getNumberOfComponents<Light>(); i++){
		lightPass(world->getComponentByIndex<Light>(i));
	}

	if (world->getNumberOfComponents<Light>() > 0){
		Light* light = world->getComponentByIndex<Light>(0);

		if(getKey(input, KeyCode::O)){
			light->direction.x += 0.01f;
		}
		else if(getKey(input, KeyCode::P)){
			light->direction.x -= 0.01f;
		}
		else if(getKey(input, KeyCode::U)){
			light->position.z += 0.01f;
		}
		else if(getKey(input, KeyCode::I)){
			light->position.z -= 0.01f;
		}
	}

	if(getKeyDown(input, KeyCode::Z)){
		std::cout << "recording depth texture data" << std::endl;
		std::vector<unsigned char> data;
		data.resize(1024*1024);
		glGetTextureImage(shadowCascadeDepth[0], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("shadow_depth_data0.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[1], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("shadow_depth_data1.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[2], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("shadow_depth_data2.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[3], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("shadow_depth_data3.bmp", data, 1024, 1024, 1);
		glGetTextureImage(shadowCascadeDepth[4], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("shadow_depth_data4.bmp", data, 1024, 1024, 1);


		glGetTextureImage(shadowSpotlightDepth, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 1024*1024*1, &data[0]);
		Util::writeToBMP("spotlight_shadow_depth_data.bmp", data, 1024, 1024, 1);
	}

	if(getKeyDown(input, KeyCode::X)){
		std::cout << "recording position and normal texture data" << std::endl;
		std::vector<unsigned char> data;
		data.resize(1024*1024*4);
		glGetTextureImage(position, 0, GL_RGBA, GL_UNSIGNED_BYTE, 1024*1024*4, &data[0]);
		Util::writeToBMP("position.bmp", data, 1024, 1024, 4);
		glGetTextureImage(normal, 0, GL_RGBA, GL_UNSIGNED_BYTE, 1024*1024*4, &data[0]);
		Util::writeToBMP("normal.bmp", data, 1024, 1024, 4);
	}

	// if(getKeyDown(input, KeyCode::C)){
	// 	std::cout << "recording cube depth texture data" << std::endl;
	// 	std::vector<unsigned char> data;
	// 	data.resize(1024*1024*1);

	// 	for(size_t i = 0; i < data.size(); i++){
	// 		if(i % 3 == 0){
	// 			data[i] = 255;
	// 		}
	// 		data[i] = 0;
	// 	}

	// 	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowCubemapDepth);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Positive_X.bmp", data, 1024, 1024, 1);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Negative_X.bmp", data, 1024, 1024, 1);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Positive_Y.bmp", data, 1024, 1024, 1);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Negative_Y.bmp", data, 1024, 1024, 1);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Positive_Z.bmp", data, 1024, 1024, 1);
	// 	glGetTexImage(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	Util::writeToBMP("cubemap_Negative_Z.bmp", data, 1024, 1024, 1);
	// 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	// }

	if(world->debug){
		debugPass();
	}

	endFrame(color);

    Graphics::checkError();
}

void ForwardRenderer::addToRenderObjectsList(MeshRenderer* meshRenderer)
{
	Transform* transform = meshRenderer->getComponent<Transform>(world);
	Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

	int transformIndex = world->getIndexOf(transform->componentId);
	int meshIndexInBuffer = meshBuffer.getIndex(meshRenderer->meshId); 

	std::cout << "" << std::endl;
	for(int i = 0; i < 8; i++){
		if(meshRenderer->materialIds[i] == Guid::INVALID){
			break;
		}

		int materialIndex = world->getIndexOfAsset(meshRenderer->materialIds[i]);
		int subMeshVertexStartIndex = mesh->subMeshVertexStartIndices[i];
		int subMeshVertexEndIndex = mesh->subMeshVertexStartIndices[i + 1];
		int subMeshVerticesCount = subMeshVertexEndIndex - subMeshVertexStartIndex;

		Material* material = world->getAssetByIndex<Material>(materialIndex);
		Shader* shader = world->getAsset<Shader>(material->shaderId);

		RenderObject renderObject;
		renderObject.id = meshRenderer->componentId;
		renderObject.start = meshBuffer.start[meshIndexInBuffer] + subMeshVertexStartIndex;
		renderObject.size = subMeshVerticesCount;
		renderObject.transformIndex = transformIndex;
		renderObject.materialIndex = materialIndex;

		std::cout << "mesh id: " << meshRenderer->meshId.toString() << " start: " << renderObject.start << " size: " << renderObject.size << " subMeshVertexStartIndex: " << subMeshVertexStartIndex << " subMeshVertexEndIndex: " << subMeshVertexEndIndex << std::endl;

		for(int j = 0; j < 10; j++){
			renderObject.shaders[j] = shader->programs[j].handle;
		}

		renderObject.mainTexture = -1;
		renderObject.normalMap = -1;
		renderObject.specularMap = -1;

		Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
		Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
		Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);

		if(mainTexture != NULL){ renderObject.mainTexture = mainTexture->handle.handle; }
		if(normalMap != NULL){ renderObject.normalMap = normalMap->handle.handle; }
		if(specularMap != NULL){ renderObject.specularMap = specularMap->handle.handle; }

		renderObject.boundingSphere = meshBuffer.boundingSpheres[meshIndexInBuffer];

		renderObjects.push_back(renderObject);	
	}
}

void ForwardRenderer::removeFromRenderObjectsList(MeshRenderer* meshRenderer)
{
	//mmm his is slow...need a faster way of removing render objects
	for(size_t i = 0; i < renderObjects.size(); i++){
		if(meshRenderer->componentId == renderObjects[i].id){
			renderObjects.erase(renderObjects.begin() + i);
		}
	}
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
	updateCameraUniformState(camera);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(camera->viewport.x, camera->viewport.y, camera->viewport.width, camera->viewport.height);
	glClearColor(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);

	// fill position, normal, and depth buffers
	// glBindVertexArray(meshBuffer.vao);

	// for(int i = 0; i < renderObjects.size(); i++){
	// 	Graphics::render(world, &mainShader, ShaderVariant::None, renderObjects[i].model, renderObjects[i].start, renderObjects[i].size, &query);
	// }

	// glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

void ForwardRenderer::cullingPass()
{
	// cull render objects based on camera frustrum
	for(size_t i = 0; i < renderObjects.size(); i++){
		//std::cout << "render object id: " << renderObjects[i].id.toString() << " transform index: " << renderObjects[i].transformIndex << std::endl;
		Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();

		glm::vec4 temp = renderObjects[i].model * glm::vec4(renderObjects[i].boundingSphere.centre, 1.0f);
		glm::vec3 centre = glm::vec3(temp.x, temp.y, temp.z);
		float radius = renderObjects[i].boundingSphere.radius;

		//if(camera->checkSphereInFrustum(centre, radius)){
		//	std::cout << "Render object inside camera frustrum " << centre.x << " " << centre.y << " " << centre.z << " " << radius << std::endl;
		//}
	}
}

void ForwardRenderer::lightPass(Light* light)
{
	LightType lightType = light->lightType;
	ShadowType shadowType = light->shadowType;
	ShaderVariant variant = ShaderVariant::None;

	//std::cout << "id: " << light->componentId.toString() << " light type: " << (int)lightType << " shadow type: " << (int)shadowType << " variant: " << (int)variant << std::endl;

	if(lightType == LightType::Directional){
		if(shadowType == ShadowType::Hard){
			variant = ShaderVariant::Directional_Hard;
		}
		else if(shadowType == ShadowType::Soft){
			variant = ShaderVariant::Directional_Soft;
		}

		calcShadowmapCascades(camera->frustum.nearPlane, camera->frustum.farPlane);
		calcCascadeOrthoProj(camera->getViewMatrix(), light->direction);

		for(int i = 0; i < 5; i++){
			glBindFramebuffer(GL_FRAMEBUFFER, shadowCascadeFBO[i]);
			glBindVertexArray(meshBuffer.vao);

			glClearDepth(1.0f);
			glClear(GL_DEPTH_BUFFER_BIT);

			GLuint shaderProgram = depthShader.programs[ShaderVariant::None].handle;

			Graphics::use(shaderProgram);
			Graphics::setMat4(shaderProgram, "view", cascadeLightView[i]);
			Graphics::setMat4(shaderProgram, "projection", cascadeOrthoProj[i]);

			for(int j = 0; j < renderObjects.size(); j++){
				Graphics::setMat4(shaderProgram, "model", renderObjects[j].model);
				Graphics::render(world, renderObjects[j], &query);
			}

			glBindVertexArray(0);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	}
	else if(lightType == LightType::Spot){
		if(shadowType == ShadowType::Hard){
			variant = ShaderVariant::Spot_Hard;
		}
		else if(shadowType == ShadowType::Soft){
			variant = ShaderVariant::Spot_Soft;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, shadowSpotlightFBO);
		glBindVertexArray(meshBuffer.vao);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		GLuint shaderProgram = depthShader.programs[ShaderVariant::None].handle;

		shadowProjMatrix = light->projection;
		shadowViewMatrix = glm::lookAt(light->position, light->position + light->direction, glm::vec3(0.0f, 1.0f, 0.0f));

		Graphics::use(shaderProgram);
		Graphics::setMat4(shaderProgram, "projection", shadowProjMatrix);
		Graphics::setMat4(shaderProgram, "view", shadowViewMatrix);

		for(int i = 0; i < renderObjects.size(); i++){
			Graphics::setMat4(shaderProgram, "model", renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindVertexArray(0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	else if(lightType == LightType::Point){
		if(shadowType == ShadowType::Hard){
			variant = ShaderVariant::Point_Hard;
		}
		else if(shadowType == ShadowType::Soft){
			variant = ShaderVariant::Point_Soft;
		}

		calcCubeViewMatrices(light->position, light->projection);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowCubemapFBO);
		glBindVertexArray(meshBuffer.vao);

		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		GLuint shaderProgram = depthCubemapShader.programs[ShaderVariant::None].handle;

		Graphics::use(shaderProgram);
		Graphics::setVec3(shaderProgram, "lightPos", light->position);
		Graphics::setFloat(shaderProgram, "farPlane", camera->frustum.farPlane);
		for(int i = 0; i < 6; i++){
			Graphics::setMat4(shaderProgram, "cubeViewProjMatrices[" + std::to_string(i) + "]", cubeViewProjMatrices[i]);
		}

		for(int i = 0; i < renderObjects.size(); i++){
			Graphics::setMat4(shaderProgram, "model", renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
		}

		glBindVertexArray(0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	updateLightUniformState(light);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glBindVertexArray(meshBuffer.vao);

	for(int i = 0; i < renderObjects.size(); i++){
		GLuint shaderProgram = renderObjects[i].shaders[(int)variant];
		Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);

		Graphics::use(shaderProgram);
		Graphics::use(shaderProgram, material, renderObjects[i]);
		Graphics::setMat4(shaderProgram, "model", renderObjects[i].model);

		if(lightType == LightType::Directional){
			for(int j = 0; j < 5; j++){
				Graphics::setInt(shaderProgram, "shadowMap[" + std::to_string(j) + "]", 3 + j);

				glActiveTexture(GL_TEXTURE0 + 3 + j);
				glBindTexture(GL_TEXTURE_2D, shadowCascadeDepth[j]);
			}
		}
		else if(lightType == LightType::Spot){
			Graphics::setInt(shaderProgram, "shadowMap[0]", 3);

			glActiveTexture(GL_TEXTURE0 + 3);
			glBindTexture(GL_TEXTURE_2D, shadowSpotlightDepth);
		}

		Graphics::render(world, renderObjects[i], &query);
	}

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	pass++;

	Graphics::checkError();
}

void ForwardRenderer::debugPass()
{		
	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	int view = world->debugView;

	glBindFramebuffer(GL_FRAMEBUFFER, debug.fbo[view].handle);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(view == 0 || view == 1 || view == 2){
		glBindVertexArray(meshBuffer.vao);

		GLuint shaderProgram = debug.shaders[view].programs[(int)ShaderVariant::None].handle;

		Graphics::use(shaderProgram);

		for(int i = 0; i < renderObjects.size(); i++){
			Graphics::setMat4(shaderProgram, "model", renderObjects[i].model);
			Graphics::render(world, renderObjects[i], &query);
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

	mainShader.vertexShader = Shader::mainVertexShader;
	mainShader.fragmentShader = Shader::mainFragmentShader;
	mainShader.compile();

	mainShader.setUniformBlock("CameraBlock", 0);

	ssaoShader.vertexShader = Shader::ssaoVertexShader;
	ssaoShader.fragmentShader = Shader::ssaoFragmentShader;
	ssaoShader.compile();

	depthShader.vertexShader = Shader::shadowDepthMapVertexShader;
	depthShader.fragmentShader = Shader::shadowDepthMapFragmentShader;
	depthShader.compile();

	depthCubemapShader.vertexShader = Shader::shadowDepthCubemapVertexShader;
	//depthCubemapShader.geometryShader = Shader::shadowDepthCubemapGeometryShader;
	depthCubemapShader.fragmentShader = Shader::shadowDepthCubemapFragmentShader;
	depthCubemapShader.compile();

	quadShader.vertexShader = Shader::windowVertexShader;
	quadShader.fragmentShader = Shader::windowFragmentShader;
	quadShader.compile();

	std::cout << "DEPTH VERTEX: " << depthShader.vertexShader << std::endl;
	std::cout << "FRAGMENT VERTEX: " << depthShader.fragmentShader << std::endl;
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

		//std::cout << "adding mesh: " << mesh->assetId.toString() << " start index: " << startIndex << std::endl;

		meshBuffer.meshIds.push_back(mesh->assetId);
		meshBuffer.start.push_back(startIndex);
		meshBuffer.count.push_back((int)mesh->vertices.size());
		meshBuffer.vertices.insert(meshBuffer.vertices.end(), mesh->vertices.begin(), mesh->vertices.end());
		meshBuffer.normals.insert(meshBuffer.normals.end(), mesh->normals.begin(), mesh->normals.end());
		meshBuffer.texCoords.insert(meshBuffer.texCoords.end(), mesh->texCoords.begin(), mesh->texCoords.end());
		meshBuffer.boundingSpheres.push_back(mesh->getBoundingSphere());

		startIndex += (int)mesh->vertices.size();
	}

	glBindVertexArray(meshBuffer.vao);
	glBindBuffer(GL_ARRAY_BUFFER, meshBuffer.vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, meshBuffer.vertices.size()*sizeof(float), &meshBuffer.vertices[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, meshBuffer.vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, meshBuffer.normals.size()*sizeof(float), &meshBuffer.normals[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, meshBuffer.vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, meshBuffer.texCoords.size()*sizeof(float), &meshBuffer.texCoords[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);

	Graphics::checkError();

	//std::cout << "mesh buffer size: " << meshBuffer.vertices.size() << " " << meshBuffer.normals.size() << " " << meshBuffer.texCoords.size() << std::endl;
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

	glGenTextures(1, &position);
	glBindTexture(GL_TEXTURE_2D, position);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera->viewport.width, camera->viewport.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &normal);
	glBindTexture(GL_TEXTURE_2D, normal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera->viewport.width, camera->viewport.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, camera->viewport.width, camera->viewport.height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, camera->viewport.width, camera->viewport.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, position, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, normal, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	glDrawBuffers(3, attachments);

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

void ForwardRenderer::createSSAOFBO()
{
	glGenFramebuffers(1, &ssaoFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);

	glGenTextures(1, &ssaoColor);
	glBindTexture(GL_TEXTURE_2D, ssaoColor);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, camera->viewport.width, camera->viewport.height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	  
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColor, 0);  

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ForwardRenderer::createShadowMapFBOs()
{
	// create directional light cascade shadow map fbo
	glGenFramebuffers(5, &shadowCascadeFBO[0]);
	glGenTextures(5, &shadowCascadeDepth[0]);

	for(int i = 0; i < 5; i++){
		glBindFramebuffer(GL_FRAMEBUFFER, shadowCascadeFBO[i]);
		glBindTexture(GL_TEXTURE_2D, shadowCascadeDepth[i]);
		// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

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
	// glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

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

void ForwardRenderer::initRenderObjectsList()
{
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		if(meshRenderer != NULL && !meshRenderer->isStatic){
			addToRenderObjectsList(meshRenderer);
		}
	}
}

void ForwardRenderer::updateRenderObjectsList()
{
	int meshRendererInstanceType = Component::getInstanceType<MeshRenderer>();
	int transformInstanceType = Component::getInstanceType<Transform>();

	// add created mesh renderers to render object list
	std::vector<Guid> meshRendererIdsAdded;

	std::vector<triple<Guid, Guid, int>> componentIdsAdded = world->getComponentIdsMarkedCreated();
	for(size_t i = 0; i < componentIdsAdded.size(); i++){
		if(componentIdsAdded[i].third == meshRendererInstanceType){
			meshRendererIdsAdded.push_back(componentIdsAdded[i].second);
		}
	}

	for(size_t i = 0; i < meshRendererIdsAdded.size(); i++){
		int globalIndex = world->getIndexOf(meshRendererIdsAdded[i]);

		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(globalIndex);
		if(meshRenderer != NULL && !meshRenderer->isStatic){
			addToRenderObjectsList(meshRenderer);
		}
	}

	// remove destroyed mesh renderers from render objects list
	std::vector<Guid> meshRendererIdsDestroyed;

	std::vector<triple<Guid, Guid, int>> componentIdsDestroyed = world->getComponentIdsMarkedLatentDestroy();
	for(size_t i = 0; i < componentIdsDestroyed.size(); i++){
		if(componentIdsDestroyed[i].third == meshRendererInstanceType){
			meshRendererIdsDestroyed.push_back(componentIdsDestroyed[i].second);
		}
	}

	for(size_t i = 0; i < meshRendererIdsDestroyed.size(); i++){
		int globalIndex = world->getIndexOf(meshRendererIdsDestroyed[i]);

		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(globalIndex);
		if(meshRenderer != NULL && !meshRenderer->isStatic){
			removeFromRenderObjectsList(meshRenderer);
		}
	}

	// update mesh renderers that have been moved in their global arrays
	std::vector<std::pair<int, int>> transformIndicesMoved;

	std::vector<triple<Guid, int, int>> componentIdsMoved = world->getComponentIdsMarkedMoved();
	for(size_t i = 0; i < componentIdsMoved.size(); i++){
		if(componentIdsMoved[i].second == transformInstanceType){
			int oldIndex = componentIdsMoved[i].third;
			int newIndex = world->getIndexOf(componentIdsMoved[i].first);
	
			transformIndicesMoved.push_back(std::make_pair(oldIndex, newIndex));
		}
	}

	for(size_t i = 0; i < transformIndicesMoved.size(); i++){
		for(size_t j = 0; j < renderObjects.size(); j++){
			if(transformIndicesMoved[i].first == renderObjects[j].transformIndex){
				renderObjects[j].transformIndex = transformIndicesMoved[i].second;
			}
		}
	}
}

void ForwardRenderer::calcShadowmapCascades(float nearDist, float farDist)
{
	const float splitWeight = 0.95f;
    const float ratio = farDist / nearDist;

    for(int i = 0; i < 6; i++){
    	const float si = i / 5.0f;

    	cascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) + (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
    }
}

void ForwardRenderer::calcCascadeOrthoProj(glm::mat4 view, glm::vec3 direction)
{
	glm::mat4 viewInv = glm::inverse(view);
	float fov = camera->frustum.fov;
	float aspect = camera->viewport.getAspectRatio();
	float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
	float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

	for (unsigned int i = 0; i < 5; i++){
		float xn = -1.0f * cascadeEnds[i] * tanHalfHFOV;
		float xf = -1.0f * cascadeEnds[i + 1] * tanHalfHFOV;
		float yn = -1.0f * cascadeEnds[i] * tanHalfVFOV;
		float yf = -1.0f * cascadeEnds[i + 1] * tanHalfVFOV;

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
		float d = 40.0f;//cascadeEnds[i + 1] - cascadeEnds[i];

		glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d*direction.x, frustrumCentreWorldSpace.y + d*direction.y, frustrumCentreWorldSpace.z + d*direction.z);

		cascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace.x, frustrumCentreWorldSpace.y, frustrumCentreWorldSpace.z), glm::vec3(1.0f, 0.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();
		for (unsigned int j = 0; j < 8; j++){
			// Transform the frustum coordinate from view to world space
			glm::vec4 vW = viewInv * frustumCorners[j];

			//std::cout << "j: " << j << " " << vW.x << " " << vW.y << " " << vW.z << " " << vW.w << std::endl;

			// Transform the frustum coordinate from world to light space
			glm::vec4 vL = cascadeLightView[i] * vW;

			//std::cout << "j: " << j << " " << vL.x << " " << vL.y << " " << vL.z << " " << vL.w << std::endl;

			minX = glm::min(minX, vL.x);
			maxX = glm::max(maxX, vL.x);
			minY = glm::min(minY, vL.y);
			maxY = glm::max(maxY, vL.y);
			minZ = glm::min(minZ, vL.z);
			maxZ = glm::max(maxZ, vL.z);
		}

		// std::cout << "i: " << i << " " << minX << " " << maxX << " " << minY << " " << maxY << " " << minZ << " " << maxZ << "      " << p.x << " " << p.y << " " << p.z << "      " << frustrumCentreWorldSpace.x << " " << frustrumCentreWorldSpace.y << " " << frustrumCentreWorldSpace.z << std::endl;

		cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
	}


	// glm::vec4 test(1.0f, 1.0f, 1.0f,1.0f);
	// glm::vec4 test2(-1.0f, -1.0f, -1.0f, 1.0f);
	// glm::mat4 testingMatrix = glm::inverse(camera->getProjMatrix() * camera->getViewMatrix());

	// test = testingMatrix*test;
	// test2 = testingMatrix*test2;

	// test2.x /= test2.w;
	// test2.y /= test2.w;
	// test2.z /= test2.w;

	// test.x /= test.w;
	// test.y /= test.w;
	// test.z /= test.w;

	// std::cout << "test: " << test.x << " " << test.y << " " << test.z << " test2: " << test2.x << " " << test2.y << " " << test2.z << std::endl;

	// test = camera->getViewMatrix() * test;
	// test2 = camera->getViewMatrix() * test2;

	// std::cout << "test: " << test.x << " " << test.y << " " << test.z << " test2: " << test2.x << " " << test2.y << " " << test2.z << std::endl;

	// glm::mat4 test = cascadeOrthoProj[0] * cascadeLightView[0];
	// std::cout << test[0][0] << " " << test[0][1] << " " << test[0][2] << " " << test[0][3] << std::endl;
	// std::cout << test[1][0] << " " << test[1][1] << " " << test[1][2] << " " << test[1][3] << std::endl;
	// std::cout << test[2][0] << " " << test[2][1] << " " << test[2][2] << " " << test[2][3] << std::endl;
	// std::cout << test[3][0] << " " << test[3][1] << " " << test[3][2] << " " << test[3][3] << std::endl;
}

void ForwardRenderer::calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection)
{
	cubeViewProjMatrices[0] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewProjMatrices[1] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewProjMatrices[2] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
	cubeViewProjMatrices[3] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
	cubeViewProjMatrices[4] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewProjMatrices[5] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
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

void ForwardRenderer::updateLightUniformState(Light* light)
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

	if(light->lightType == LightType::Directional){
		for(int i = 0; i < 5; i++){
			lightState.lightProjection[i] = cascadeOrthoProj[i];

			glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, cascadeEnds[i + 1], 1.0f);
			glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
			lightState.cascadeEnds[i] = clipSpaceCascadeEnd.z;

			lightState.lightView[i] = cascadeLightView[i];
		}
	}
	else if(light->lightType == LightType::Spot){
		for(int i = 0; i < 5; i++){
			lightState.lightProjection[i] = shadowProjMatrix;
			lightState.lightView[i] = shadowViewMatrix;
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
}