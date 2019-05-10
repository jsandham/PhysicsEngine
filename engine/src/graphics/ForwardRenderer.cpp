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

	depthShader.vertexShader = Shader::depthMapVertexShader;
	depthShader.fragmentShader = Shader::depthMapFragmentShader;
	depthShader.compile();

	quadShader.vertexShader = Shader::windowVertexShader;
	quadShader.fragmentShader = Shader::windowFragmentShader;
	quadShader.compile();

	// generate graphics mesh buffers so we can draw dynamic meshes
	// for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
	// 	Mesh* mesh = world->getAssetByIndex<Mesh>(i);

	// 	Guid meshId = mesh->assetId;

	// 	InternalMesh internalMesh;

	// 	size_t numVertices = mesh->vertices.size() / 3;
	// 	size_t numNormals = mesh->normals.size() / 3;
	// 	size_t numTexCoords = mesh->texCoords.size() / 2;

	// 	glGenVertexArrays(1, &internalMesh.VAO);
	// 	glBindVertexArray(internalMesh.VAO);

	// 	glGenBuffers(1, &internalMesh.vertexVBO);
	// 	glBindBuffer(GL_ARRAY_BUFFER, internalMesh.vertexVBO);
	// 	glBufferData(GL_ARRAY_BUFFER, numVertices*sizeof(float), &(mesh->vertices[0]), GL_DYNAMIC_DRAW);
	// 	glEnableVertexAttribArray(0);
	// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	// 	glGenBuffers(1, &internalMesh.normalVBO);
	// 	glBindBuffer(GL_ARRAY_BUFFER, internalMesh.normalVBO);
	// 	glBufferData(GL_ARRAY_BUFFER, numNormals*sizeof(float), &(mesh->normals[0]), GL_DYNAMIC_DRAW);
	// 	glEnableVertexAttribArray(1);
	// 	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	// 	glGenBuffers(1, &internalMesh.texCoordVBO);
	// 	glBindBuffer(GL_ARRAY_BUFFER, internalMesh.texCoordVBO);
	// 	glBufferData(GL_ARRAY_BUFFER, numTexCoords*sizeof(float), &(mesh->texCoords[0]), GL_DYNAMIC_DRAW);
	// 	glEnableVertexAttribArray(2);
	// 	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	// 	glBindVertexArray(0);

	// 	std::map<Guid, InternalMesh>::iterator it = meshIdToInternalMesh.find(meshId);
	// 	if(it == meshIdToInternalMesh.end()){  // do I even really need this check?
	// 		meshIdToInternalMesh[meshId] = internalMesh;
	// 	}
	// 	else{
	// 		std::cout << "Error: Duplicate mesh found" << std::endl;  
	// 		return;
	// 	}
	// }

	// generate one large vertex buffer storing all unique mesh vertices, normals, and tex coordinates
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

	// grab camera
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	// generate fbo
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glGenTextures(1, &color);
	glBindTexture(GL_TEXTURE_2D, color);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera->width, camera->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, camera->width, camera->height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[1] = { GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, attachments);

	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
		std::cout << "ERROR: FRAMEBUFFER IS NOT COMPLETE " << framebufferStatus << std::endl;
	}
	
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
	initDirectionalLightUniformState();
	initSpotLightUniformState();
	initPointLightUniformState();

	debug.init();

	Graphics::checkError();
}

void ForwardRenderer::update()
{
	if(camera == NULL) { return; }

	query.numBatchDrawCalls = 0;
	query.numDrawCalls = 0;
	query.totalElapsedTime = 0.0f;
	query.verts = 0;
	query.tris = 0;
	query.lines = 0;
	query.points = 0;

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glViewport(camera->x, camera->y, camera->width, camera->height - 40);

	glm::vec4 backColor = camera->getBackgroundColor();
	glClearColor(backColor.x, backColor.y, backColor.z, backColor.w);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	updateCameraUniformState();

	// cull render objects based on camera frustrum
	for(size_t i = 0; i < renderObjects.size(); i++){
		Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();

		glm::vec4 temp = renderObjects[i].model * glm::vec4(renderObjects[i].boundingSphere.centre, 1.0f);
		glm::vec3 centre = glm::vec3(temp.x, temp.y, temp.z);
		float radius = renderObjects[i].boundingSphere.radius;

		// if(camera->checkSphereInFrustum(centre, radius)){
		// 	std::cout << "Render object inside camera frustrum " << centre.x << " " << centre.y << " " << centre.z << " " << radius << std::endl;
		// }
	}

	// perform depth pass 
	// Graphics::use(&depthShader);
	// for(size_t i = 0; i < renderObjects.size(); i++){

	// }



	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_ONE, GL_ZERO);
	glBlendEquation(GL_FUNC_ADD);

	pass = 0;

	if (world->getNumberOfComponents<DirectionalLight>() > 0){
		DirectionalLight* directionalLight = world->getComponentByIndex<DirectionalLight>(0);

		updateDirectionalLightUniformState(directionalLight);
		//calcCascadeOrthoProj(directionalLight->direction);

		//render();

		pass++;
	}

	for(int i = 0; i < world->getNumberOfComponents<SpotLight>(); i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		SpotLight* spotLight = world->getComponentByIndex<SpotLight>(i);

		updateSpotLightUniformState(spotLight);

		//render();

		pass++;
	}

	for(int i = 0; i < world->getNumberOfComponents<PointLight>(); i++){
		if(pass >= 1){ glBlendFunc(GL_ONE, GL_ONE); }

		PointLight* pointLight = world->getComponentByIndex<PointLight>(i);

		updatePointLightUniformState(pointLight);

		//render();
			
		pass++;
	}

	if(world->debug){
		renderDebug(world->debugView);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// render to quad
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Graphics::use(&quadShader);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color);

    glBindVertexArray(quadVAO);
    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    Graphics::checkError();
}

void ForwardRenderer::sort()
{

}

void ForwardRenderer::add(MeshRenderer* meshRenderer)
{
	if(meshRenderer != NULL && !meshRenderer->isStatic){
		Transform* transform = meshRenderer->getComponent<Transform>(world);

		int transformIndex = world->getIndexOf(transform->componentId);
		int materialIndex = world->getIndexOfAsset(meshRenderer->materialId);

		int index = meshBuffer.getIndex(meshRenderer->meshId);

		std::cout << "transform index: " << transformIndex << " " << materialIndex << " mesh id: " << meshRenderer->meshId.toString() << " index: " << index << std::endl;

		std::cout << "index: " << index << "  " << meshBuffer.start[index] << " " << meshBuffer.count[index] << "  " << transformIndex << " " << materialIndex << std::endl;

		if(transformIndex != -1 && materialIndex != -1 && index != -1){
			RenderObject renderObject;
			renderObject.start = meshBuffer.start[index];
			renderObject.size = meshBuffer.count[index];
			renderObject.transformIndex = transformIndex;
			renderObject.materialIndex = materialIndex;
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

void ForwardRenderer::render()
{
	batchManager.render(world, &query);

	glBindVertexArray(meshBuffer.vao);

	for(int i = 0; i < renderObjects.size(); i++){
		Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);

		Graphics::render(world, material, renderObjects[i].model, renderObjects[i].start, renderObjects[i].size, &query);
	}

	glBindVertexArray(0);


	// for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
	// 	MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

	// 	if(meshRenderer != NULL && !meshRenderer->isStatic){
	// 		Transform* transform = meshRenderer->getComponent<Transform>(world);
	// 		Material* material = world->getAsset<Material>(meshRenderer->materialId);
	// 		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

	// 		glm::mat4 model = transform->getModelMatrix();

	// 		// get internal mesh and render
	// 		std::map<Guid,InternalMesh>::iterator it = meshIdToInternalMesh.find(mesh->assetId);
	// 		if(it != meshIdToInternalMesh.end()){
	// 			InternalMesh internalMesh = it->second;

	// 			Graphics::render(world, material, model, internalMesh.VAO, (int)mesh->vertices.size() / 3, &query);
	// 		}

	// 	}
	// }
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
		batchManager.render(world, &debug.shaders[view], NULL);

		//render non static meshes here
		glBindVertexArray(meshBuffer.vao);

		for(int i = 0; i < renderObjects.size(); i++){
			Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);

			Graphics::render(world, &debug.shaders[view], renderObjects[i].model, renderObjects[i].start, renderObjects[i].size, &query);
		}

		glBindVertexArray(0);

		// for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		// 	MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

		// 	if(meshRenderer != NULL && !meshRenderer->isStatic){
		// 		Transform* transform = meshRenderer->getComponent<Transform>(world);
		// 		Material* material = world->getAsset<Material>(meshRenderer->materialId);
		// 		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

		// 		glm::mat4 model = transform->getModelMatrix();

		// 		// get internal mesh and render
		// 		std::map<Guid,InternalMesh>::iterator it = meshIdToInternalMesh.find(mesh->assetId);
		// 		if(it != meshIdToInternalMesh.end()){
		// 			InternalMesh internalMesh = it->second;

		// 			Graphics::render(world, material, model, internalMesh.VAO, (int)mesh->vertices.size() / 3, &query);
		// 		}

		// 	}
		// }
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ForwardRenderer::initCameraUniformState()
{
	glGenBuffers(1, &(cameraState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraState.handle, 0, 144);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::initDirectionalLightUniformState()
{
	glGenBuffers(1, &(directionLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, directionLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, directionLightState.handle, 0, 64);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::initSpotLightUniformState()
{
	glGenBuffers(1, &(spotLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, spotLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 100, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 2, spotLightState.handle, 0, 100);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::initPointLightUniformState()
{
	glGenBuffers(1, &(pointLightState.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, pointLightState.handle);
	glBufferData(GL_UNIFORM_BUFFER, 76, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 3, pointLightState.handle, 0, 76);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateCameraUniformState()
{
	cameraState.projection = camera->getProjMatrix();
	cameraState.view = camera->getViewMatrix();
	cameraState.cameraPos = camera->getPosition();

	glBindBuffer(GL_UNIFORM_BUFFER, cameraState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(cameraState.projection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(cameraState.view));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 16, glm::value_ptr(cameraState.cameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateDirectionalLightUniformState(DirectionalLight* light)
{
	directionLightState.direction = light->direction;
	directionLightState.ambient = light->ambient;
	directionLightState.diffuse = light->diffuse;
	directionLightState.specular = light->specular;

	glBindBuffer(GL_UNIFORM_BUFFER, directionLightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(directionLightState.direction));
	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(directionLightState.ambient));
	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(directionLightState.diffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(directionLightState.specular));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::updateSpotLightUniformState(SpotLight* light)
{
	spotLightState.position = light->position;
	spotLightState.direction = light->direction;
	spotLightState.ambient = light->ambient;
	spotLightState.diffuse = light->diffuse;
	spotLightState.specular = light->specular;
	spotLightState.constant = light->constant;
	spotLightState.linear = light->linear;
	spotLightState.quadratic = light->quadratic;
	spotLightState.cutOff = light->cutOff;
	spotLightState.outerCutOff = light->outerCutOff;

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
}

void ForwardRenderer::updatePointLightUniformState(PointLight* light)
{
	pointLightState.position = light->position;
	pointLightState.ambient = light->ambient;
	pointLightState.diffuse = light->diffuse;
	pointLightState.specular = light->specular;
	pointLightState.constant = light->constant;
	pointLightState.linear = light->linear;
	pointLightState.quadratic = light->quadratic;

	glBindBuffer(GL_UNIFORM_BUFFER, pointLightState.handle);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(pointLightState.position));
	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(pointLightState.ambient));
	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(pointLightState.diffuse));
	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(pointLightState.specular));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 4, &pointLightState.constant);
	glBufferSubData(GL_UNIFORM_BUFFER, 68, 4, &pointLightState.linear);
	glBufferSubData(GL_UNIFORM_BUFFER, 72, 4, &pointLightState.quadratic);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void ForwardRenderer::createShadowMapTextures()
{

}