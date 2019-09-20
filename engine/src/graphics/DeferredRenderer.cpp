#include"../../include/graphics/DeferredRenderer.h"

using namespace PhysicsEngine;

DeferredRenderer::DeferredRenderer()
{

}

DeferredRenderer::~DeferredRenderer()
{

}

void DeferredRenderer::init(World* world)
{
	std::cout << "Deferred Renderer init called" << std::endl;

	this->world = world;

	glGenQueries(1, &(query.queryId));

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

			if(!shader->isCompiled){
				std::cout << "Shader failed to compile " << i << " " << shader->assetId.toString() << std::endl;
			}

			std::string uniformBlocks[] = {"CameraBlock", 
										   "DirectionalLightBlock", 
										   "SpotLightBlock", 
										   "PointLightBlock"};

			for(int j = 0; j < 4; j++){
				shader->setUniformBlock(uniformBlocks[j], j);
			}
		}
	}

	// generate one large vertex buffer storing all mesh vertices, normals, and tex coordinates
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

	std::cout << "mesh buffer size: " << meshBuffer.vertices.size() << " " << meshBuffer.normals.size() << " " << meshBuffer.texCoords.size() << std::endl;

	// generate render objects list
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		if(meshRenderer != NULL){
			add(meshRenderer);
		}
	}

	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	int width = camera->viewport.width;
	int height = camera->viewport.height;

	// generate G-Buffer
	glGenFramebuffers(1, &gbuffer.handle);
	glBindFramebuffer(GL_FRAMEBUFFER, gbuffer.handle);

	glGenTextures(1, &gbuffer.color0);
	glBindTexture(GL_TEXTURE_2D, gbuffer.color0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &gbuffer.color1);
	glBindTexture(GL_TEXTURE_2D, gbuffer.color1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &gbuffer.color2);
	glBindTexture(GL_TEXTURE_2D, gbuffer.color2);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &gbuffer.depth);
	glBindTexture(GL_TEXTURE_2D, gbuffer.depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gbuffer.color0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gbuffer.color1, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gbuffer.color2, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gbuffer.depth, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);

	if ((gbuffer.gBufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
		std::cout << "ERROR: GBUFFER IS NOT COMPLETE " << gbuffer.gBufferStatus << std::endl;
	}
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	gbuffer.shader.vertexShader = Shader::gbufferVertexShader;
	gbuffer.shader.fragmentShader = Shader::gbufferFragmentShader;

	gbuffer.shader.compile();

	Graphics::checkError();
	// GLenum error;
	// while ((error = glGetError()) != GL_NO_ERROR){
	// 	std::cout << "Error: Deferred Renderer failed with error code: " << error << " during init" << std::endl;
	// }

	std::cout << "Framebuffer : " << gbuffer.handle << std::endl;
}

void DeferredRenderer::update(Input input)
{
	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	glm::mat4 projection = camera->getProjMatrix();
	glm::mat4 view = camera->getViewMatrix();

	// geometry pass
	glBindFramebuffer(GL_FRAMEBUFFER, gbuffer.handle);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Graphics::use(&gbuffer.shader, ShaderVariant::None);
	gbuffer.shader.setMat4("projection", ShaderVariant::None, projection);
	gbuffer.shader.setMat4("view", ShaderVariant::None, view);

	std::cout << "number of render objects: " << renderObjects.size() << std::endl;

	for(size_t i = 0; i < renderObjects.size(); i++){
		Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

		renderObjects[i].model = transform->getModelMatrix();
	}

	glBindVertexArray(meshBuffer.vao);

	for(size_t i = 0; i < renderObjects.size(); i++){
		Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);

		gbuffer.shader.setMat4("model", ShaderVariant::None, renderObjects[i].model);

		GLsizei numVertices = renderObjects[i].size / 3;
		GLint startIndex = renderObjects[i].start / 3;

		glDrawArrays(GL_TRIANGLES, startIndex, numVertices);
	}

	glBindVertexArray(0);

	// if(getKeyDown(input, KeyCode::X)){
	// 	std::cout << "XXXXXXXXXXXXXX" << std::endl;
	// 	data.resize(4 * camera->width * camera->height);

	// 	//glReadBuffer(GL_DEPTH_COMPONENT);
	// 	//glReadPixels(0, 0, camera->width, camera->height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	// 	glReadBuffer(GL_COLOR_ATTACHMENT2);
	// 	glReadPixels(0, 0, camera->width, camera->height, GL_RGBA, GL_UNSIGNED_BYTE, &data[0]);


	// 	std::cout << "Read successfull" << std::endl;

	// 	World::writeToBMP("deferred.bmp", data, camera->width, camera->height, 4);
	// }

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// lighting pass
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gbuffer.color0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gbuffer.color1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gbuffer.color2);



    Graphics::checkError();
	// GLenum error;
	// while ((error = glGetError()) != GL_NO_ERROR){
	// 	std::cout << "Error: Deferred Renderer failed with error code: " << error << " during update" << std::endl;;
	// }
}

void DeferredRenderer::sort()
{

}

void DeferredRenderer::add(MeshRenderer* meshRenderer)
{
	// if(meshRenderer != NULL && !meshRenderer->isStatic){
	// 	Transform* transform = meshRenderer->getComponent<Transform>(world);

	// 	int transformIndex = world->getIndexOf(transform->componentId);
	// 	int materialIndex = world->getIndexOfAsset(meshRenderer->materialIds[0]);

	// 	int index = meshBuffer.getIndex(meshRenderer->meshId);

	// 	std::cout << "transform index: " << transformIndex << " " << materialIndex << " mesh id: " << meshRenderer->meshId.toString() << " index: " << index << std::endl;

	// 	std::cout << "index: " << index << "  " << meshBuffer.start[index] << " " << meshBuffer.count[index] << "  " << transformIndex << " " << materialIndex << std::endl;

	// 	if(transformIndex != -1 && materialIndex != -1 && index != -1){
	// 		RenderObject renderObject;
	// 		renderObject.start = meshBuffer.start[index];
	// 		renderObject.size = meshBuffer.count[index];
	// 		renderObject.transformIndex = transformIndex;
	// 		renderObject.materialIndex = materialIndex;

	// 		renderObjects.push_back(renderObject);
	// 	}		
	// }
}

void DeferredRenderer::remove(MeshRenderer* meshRenderer)
{

}