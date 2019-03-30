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

	glGenQueries(1, &(query.queryId));

	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	// generate all texture assets
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

	// compile all shader assets and configure uniform blocks
	for(int i = 0; i < world->getNumberOfAssets<Shader>(); i++){
		Shader* shader = world->getAssetByIndex<Shader>(i);

		if(shader != NULL){
			shader->compile();

			if(!shader->isCompiled()){
				std::cout << "Shader failed to compile " << i << " " << shader->assetId.toString() << std::endl;
			}

			std::string uniformBlocks[] = {"CameraBlock", 
										   "DirectionalLightBlock", 
										   "SpotLightBlock", 
										   "PointLightBlock"};

			for(int i = 0; i < 4; i++){
				GLuint blockIndex = glGetUniformBlockIndex(shader->program.handle, uniformBlocks[i].c_str()); 
				if (blockIndex != GL_INVALID_INDEX){
					glUniformBlockBinding(shader->program.handle, blockIndex, i);
				}
			}
		}
	}

	// generate graphics mesh buffers so we can draw dynamic meshes
	for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		Guid meshId = mesh->assetId;

		InternalMesh internalMesh;

		size_t numVertices = mesh->vertices.size() / 3;
		size_t numNormals = mesh->normals.size() / 3;
		size_t numTexCoords = mesh->texCoords.size() / 2;

		glGenVertexArrays(1, &internalMesh.VAO);
		glBindVertexArray(internalMesh.VAO);

		glGenBuffers(1, &internalMesh.vertexVBO);
		glBindBuffer(GL_ARRAY_BUFFER, internalMesh.vertexVBO);
		glBufferData(GL_ARRAY_BUFFER, numVertices*sizeof(float), &(mesh->vertices[0]), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

		glGenBuffers(1, &internalMesh.normalVBO);
		glBindBuffer(GL_ARRAY_BUFFER, internalMesh.normalVBO);
		glBufferData(GL_ARRAY_BUFFER, numNormals*sizeof(float), &(mesh->normals[0]), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

		glGenBuffers(1, &internalMesh.texCoordVBO);
		glBindBuffer(GL_ARRAY_BUFFER, internalMesh.texCoordVBO);
		glBufferData(GL_ARRAY_BUFFER, numTexCoords*sizeof(float), &(mesh->texCoords[0]), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

		glBindVertexArray(0);

		std::map<Guid, InternalMesh>::iterator it = meshIdToInternalMesh.find(meshId);
		if(it == meshIdToInternalMesh.end()){  // do I even really need this check?
			meshIdToInternalMesh[meshId] = internalMesh;
		}
		else{
			std::cout << "Error: Duplicate mesh found" << std::endl;  
			return;
		}
	}

	// batch all static meshes by material
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

		if(meshRenderer != NULL && meshRenderer->isStatic){
			Transform* transform = meshRenderer->getComponent<Transform>(world);
			Material* material = world->getAsset<Material>(meshRenderer->materialId);
			Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

			glm::mat4 model = transform->getModelMatrix();

			batchManager.add(material, mesh, model);
		}
	}

	glGenBuffers(1, &(cameraState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraState.handle, 0, 144);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(directionLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, directionLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, directionLightState.handle, 0, 64);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(spotLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, spotLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 100, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 2, spotLightState.handle, 0, 100);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(pointLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, pointLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 76, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 3, pointLightState.handle, 0, 76);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	debug.init();

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Forward Renderer failed with error code: " << error << " during initialization" << std::endl;;
	}
}

void ForwardRenderer::update()
{
	query.numBatchDrawCalls = 0;
	query.numDrawCalls = 0;
	query.totalElapsedTime = 0;

	int numberOfDirectionalLights = world->getNumberOfComponents<DirectionalLight>();
	int numberOfSpotLights = world->getNumberOfComponents<SpotLight>();
	int numberOfPointLights = world->getNumberOfComponents<PointLight>();

	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	glViewport(camera->x, camera->y, camera->width, camera->height - 40);

	glm::vec4 color = camera->getBackgroundColor();
	glClearColor(color.x, color.y, color.z, color.w);
	glClear(GL_COLOR_BUFFER_BIT);
	glClearDepth(1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);

	cameraState.projection = camera->getProjMatrix();
	cameraState.view = camera->getViewMatrix();
	cameraState.cameraPos = camera->getPosition();

	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(cameraState.projection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(cameraState.view));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 16, glm::value_ptr(cameraState.cameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	pass = 0;

	if (numberOfDirectionalLights > 0){
		DirectionalLight* directionalLight = world->getComponentByIndex<DirectionalLight>(0);

		directionLightState.direction = directionalLight->direction;
		directionLightState.ambient = directionalLight->ambient;
		directionLightState.diffuse = directionalLight->diffuse;
		directionLightState.specular = directionalLight->specular;

		glBindBuffer(GL_UNIFORM_BUFFER, directionLightState.handle);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(directionLightState.direction));
		glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(directionLightState.ambient));
		glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(directionLightState.diffuse));
		glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(directionLightState.specular));
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		render();

		pass++;
	}

	for(int i = 0; i < numberOfSpotLights; i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		SpotLight* spotLight = world->getComponentByIndex<SpotLight>(i);

		spotLightState.position = spotLight->position;
		spotLightState.direction = spotLight->direction;
		spotLightState.ambient = spotLight->ambient;
		spotLightState.diffuse = spotLight->diffuse;
		spotLightState.specular = spotLight->specular;
		spotLightState.constant = spotLight->constant;
		spotLightState.linear = spotLight->linear;
		spotLightState.quadratic = spotLight->quadratic;
		spotLightState.cutOff = spotLight->cutOff;
		spotLightState.outerCutOff = spotLight->outerCutOff;

		glBindBuffer(GL_UNIFORM_BUFFER, spotLightState.handle);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(spotLightState.position));
		glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(spotLightState.direction));
		glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(spotLightState.ambient));
		glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(spotLightState.diffuse));
		glBufferSubData(GL_UNIFORM_BUFFER, 64, 16, glm::value_ptr(spotLightState.specular));
		glBufferSubData(GL_UNIFORM_BUFFER, 80, 4, &(spotLightState.constant));
		glBufferSubData(GL_UNIFORM_BUFFER, 84, 4, &(spotLightState.linear));
		glBufferSubData(GL_UNIFORM_BUFFER, 88, 4, &(spotLightState.quadratic));
		glBufferSubData(GL_UNIFORM_BUFFER, 92, 4, &(spotLightState.cutOff));
		glBufferSubData(GL_UNIFORM_BUFFER, 96, 4, &(spotLightState.outerCutOff));
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		render();

		pass++;
	}

	for(int i = 0; i < numberOfPointLights; i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		PointLight* pointLight = world->getComponentByIndex<PointLight>(i);

		pointLightState.position = pointLight->position;
		pointLightState.ambient = pointLight->ambient;
		pointLightState.diffuse = pointLight->diffuse;
		pointLightState.specular = pointLight->specular;
		pointLightState.constant = pointLight->constant;
		pointLightState.linear = pointLight->linear;
		pointLightState.quadratic = pointLight->quadratic;

		glBindBuffer(GL_UNIFORM_BUFFER, pointLightState.handle);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(pointLightState.position));
		glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(pointLightState.ambient));
		glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(pointLightState.diffuse));
		glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(pointLightState.specular));
		glBufferSubData(GL_UNIFORM_BUFFER, 64, 4, &pointLightState.constant);
		glBufferSubData(GL_UNIFORM_BUFFER, 68, 4, &pointLightState.linear);
		glBufferSubData(GL_UNIFORM_BUFFER, 72, 4, &pointLightState.quadratic);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		render();
			
		pass++;
	}

	if(world->debug){
		renderDebug();
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Forward Renderer failed with error code: " << error << " during update" << std::endl;;
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

void ForwardRenderer::render()
{
	batchManager.render(world, &query);

	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

		if(meshRenderer != NULL && !meshRenderer->isStatic){
			Transform* transform = meshRenderer->getComponent<Transform>(world);
			Material* material = world->getAsset<Material>(meshRenderer->materialId);
			Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

			glm::mat4 model = transform->getModelMatrix();

			// get internal mesh and render
		}
	}
}

void ForwardRenderer::renderDebug()
{			
	for(int i = 0; i < 3; i++){
		glBindFramebuffer(GL_FRAMEBUFFER, debug.fbo[i].handle);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_DEPTH_TEST);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		batchManager.render(world, &debug.shaders[i], NULL);

		// TODO: render non static meshes here

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}