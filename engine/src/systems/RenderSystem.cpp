#include <cstddef>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

const unsigned int SHADOW_WIDTH = 1000, SHADOW_HEIGHT = 1000;
const unsigned int DEBUG_WIDTH = 1000, DEBUG_HEIGHT = 1000;

const unsigned int NUM_CASCADES = 5;

RenderSystem::RenderSystem(Manager *manager, SceneContext* context)
{
	this->manager = manager;

	numLights = 0;

	cascadeLightView.resize(NUM_CASCADES);
	cascadeOrthoProj.resize(NUM_CASCADES);

	cubeViewMatrices.resize(6);

	cascadeTexture2D.resize(6);
	shadowTexture2D = NULL;
	shadowCubemap = NULL;
	//shadowFBO = NULL;
}

RenderSystem::~RenderSystem()
{
	for (unsigned int i = 0; i < NUM_CASCADES; i++){
		delete cascadeTexture2D[i];
	}

	delete shadowTexture2D;
	delete shadowCubemap;
	//delete shadowFBO;
}


void RenderSystem::init()
{
	std::cout << "Render system init called" << std::endl;

	for(int i = 0; i < manager->getNumberOfTextures(); i++){
		Graphics::generate(manager->getTexture2D(i));
	}

	for(int i = 0; i < manager->getNumberOfShaders(); i++){
		Shader* shader = manager->getShader(i);

		shader->compile();

		if(!shader->isCompiled()){
			std::cout << "Shader failed to compile " << i << std::endl;
		}

		Graphics::setUniformBlockToBindingPoint(shader, "CameraBlock", 0);
		Graphics::setUniformBlockToBindingPoint(shader, "ShadowBlock", 1);
		Graphics::setUniformBlockToBindingPoint(shader, "DirectionalLightBlock", 2);
		Graphics::setUniformBlockToBindingPoint(shader, "SpotLightBlock", 3);
		Graphics::setUniformBlockToBindingPoint(shader, "PointLightBlock", 4);
	}

	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
	for(int i = 0; i < manager->getNumberOfMeshes(); i++){
		Mesh* mesh = manager->getMesh(i);

		Graphics::generate(mesh);
	}
	
	Graphics::generate(&cameraState);
	Graphics::generate(&shadowState);
	Graphics::generate(&directionLightState);
	Graphics::generate(&spotLightState);
	Graphics::generate(&pointLightState);

	createShadowMaps();

	Graphics::enableDepthTest();
	Graphics::enableCubemaps();
	Graphics::enablePoints();
	Graphics::checkError();
}

void RenderSystem::update()
{
	std::cout << "Render update called" << std::endl;

	int numberOfDirectionalLights = manager->getNumberOfDirectionalLights();
	int numberOfSpotLights = manager->getNumberOfSpotLights();
	int numberOfPointLights = manager->getNumberOfPointLights();

	Camera* camera;
	if(manager->getNumberOfCameras() > 0){
		camera = manager->getCamera(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	Graphics::setViewport(camera->x, camera->y, camera->width, camera->height);
	Graphics::clearColorBuffer(camera->getBackgroundColor());
	Graphics::clearDepthBuffer(1.0f);

	projection = camera->getProjMatrix();
	view = camera->getViewMatrix();
	cameraPos = camera->getPosition();

	Graphics::bind(&cameraState);
	Graphics::setProjectionMatrix(&cameraState, camera->getProjMatrix());
	Graphics::setViewMatrix(&cameraState, camera->getViewMatrix());
	Graphics::setCameraPosition(&cameraState, camera->getPosition());
	Graphics::unbind(&cameraState);

	pass = 0;

	if (numberOfDirectionalLights > 0){
		DirectionalLight* directionalLight = manager->getDirectionalLight(0);

		calcCascadeOrthoProj(directionalLight->direction);

		// 	for (unsigned int i = 0; i < NUM_CASCADES; i++){
		// 		renderShadowMap(cascadeTexture2D[i], cascadeLightView[i], cascadeOrthoProj[i]);
		// 	}

		float ends[NUM_CASCADES] = { -2.0f, -4.0f, -10.0f, -20.0f, -100.0f };

		// 	state.bind(UniformBuffer::ShadowBuffer);
		// 	for (unsigned int i = 0; i < 5; i++){
		// 		state.setLightProjectionMatrix(cascadeOrthoProj[i], i);
		// 		state.setLightViewMatrix(cascadeLightView[i], i);

		// 		glm::vec4 v(0.0f, 0.0f, ends[i], 1.0f);
		// 		glm::vec4 zClip = projection*v;

		// 		state.setCascadeEnd(zClip.z, i);
		// 	}
		// 	state.unbind(UniformBuffer::ShadowBuffer);
		Graphics::bind(&shadowState);
		for(unsigned int i = 0; i < 5; i++){
			Graphics::setLightProjectionMatrix(&shadowState, cascadeOrthoProj[i], i);
			Graphics::setLightViewMatrix(&shadowState, cascadeLightView[i], i);

			glm::vec4 v(0.0f, 0.0f, ends[i], 1.0f);
			glm::vec4 zClip = projection*v;

			Graphics::setCascadeEnd(&shadowState, zClip.z, i);
		}
		Graphics::unbind(&shadowState);

		Graphics::bind(&directionLightState);
		Graphics::setDirLightDirection(&directionLightState, -directionalLight->direction);
		Graphics::setDirLightAmbient(&directionLightState, directionalLight->ambient);
		Graphics::setDirLightDiffuse(&directionLightState, directionalLight->diffuse);
		Graphics::setDirLightSpecular(&directionLightState, directionalLight->specular);
		Graphics::unbind(&directionLightState);

		// 	state.cascadeTexture2D = cascadeTexture2D;

		renderScene();

		pass++;
	}

	for(int i = 0; i < numberOfSpotLights; i++){
		SpotLight* spotLight = manager->getSpotLight(i);

		glm::vec3 position = spotLight->position;
		glm::vec3 direction = spotLight->direction;
		glm::mat4 lightProjection = spotLight->projection;
		glm::mat4 lightView = glm::lookAt(position, position - direction, glm::vec3(0.0f, 1.0f, 0.0f));

		// renderShadowMap(shadowTexture2D, lightView, lightProjection);

		// 	state.bind(UniformBuffer::ShadowBuffer);
		// 	state.setLightProjectionMatrix(spotLights[i]->projection, 0);
		// 	state.setLightViewMatrix(lightView, 0);
		// 	state.unbind(UniformBuffer::ShadowBuffer);
		Graphics::bind(&shadowState);
		Graphics::setLightProjectionMatrix(&shadowState, spotLight->projection, 0);
		Graphics::setLightViewMatrix(&shadowState, lightView, 0);
		Graphics::unbind(&shadowState);

		Graphics::bind(&spotLightState);
		Graphics::setSpotLightPosition(&spotLightState, spotLight->position);
		Graphics::setSpotLightDirection(&spotLightState, spotLight->direction);
		Graphics::setSpotLightAmbient(&spotLightState, spotLight->ambient);
		Graphics::setSpotLightDiffuse(&spotLightState, spotLight->diffuse);
		Graphics::setSpotLightSpecular(&spotLightState, spotLight->specular);
		Graphics::setSpotLightConstant(&spotLightState, spotLight->constant);
		Graphics::setSpotLightLinear(&spotLightState, spotLight->linear);
		Graphics::setSpotLightQuadratic(&spotLightState, spotLight->quadratic);
		Graphics::setSpotLightCutoff(&spotLightState, spotLight->cutOff);
		Graphics::setSpotLightOuterCutoff(&spotLightState, spotLight->outerCutOff);
		Graphics::unbind(&spotLightState);

		// 	state.shadowTexture2D = shadowTexture2D;

		renderScene();

		pass++;
	}

	for(int i = 0; i < numberOfPointLights; i++){
		PointLight* pointLight = manager->getPointLight(i);

		glm::vec3 lightPosition = pointLight->position;
		glm::mat4 lightProjection = pointLight->projection;

		calcCubeViewMatrices(lightPosition, lightProjection);

		// 	renderDepthCubemap(shadowCubemap, lightProjection);

		Graphics::bind(&shadowState);
		Graphics::setFarPlane(&shadowState, 25.0f);
		Graphics::unbind(&shadowState);

		Graphics::bind(&pointLightState);
		Graphics::setPointLightPosition(&pointLightState, pointLight->position);
		Graphics::setPointLightAmbient(&pointLightState, pointLight->ambient);
		Graphics::setPointLightDiffuse(&pointLightState, pointLight->diffuse);
		Graphics::setPointLightSpecular(&pointLightState, pointLight->specular);
		Graphics::setPointLightConstant(&pointLightState, pointLight->constant);
		Graphics::setPointLightLinear(&pointLightState, pointLight->linear);
		Graphics::setPointLightQuadratic(&pointLightState, pointLight->quadratic);
		Graphics::unbind(&pointLightState);

		// 	state.shadowCubemap = shadowCubemap;

		renderScene();
			
		pass++;
	}

	Graphics::checkError();
}

void RenderSystem::renderScene()
{
	// for(int i = 0; i < manager->getNumberOfMeshRenderers(); i++){
	// 	MeshRenderer* meshRenderer = manager->getMeshRenderer(i);
		
	// 	Transform* transform = manager->getEntity(meshRenderer->globalEntityIndex)->getComponent<Transform>(manager->getTransforms());
	// 	// Or...
	// 	Transform* transform = manager->getComponent<Transform>(meshRenderer->globalEntityIndex);
	// 	// Or...
	// 	int transformGlobalIndex = manager->getEntity(meshRenderer->globalEntityIndex)->getComponentIndex<Transform>();
	// 	Transform* transform = manager->getTransform(transformGlobalIndex);
	// 	// Or...
	// 	Transform* transform = meshRenderer->getComponent<Transform>(manager);

	// 	Mesh* mesh = manager->getMesh(meshRenderer->meshGlobalIndex);
	// 	Material* material = manager->getMaterial(meshRenderer->materialGlobalIndex);


	// }

	// std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
	// for (unsigned int j = 0; j < meshRenderers.size(); j++){
	// 	Transform *transform = meshRenderers[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
	// 	Material *material = manager->getMaterial(meshRenderers[j]->materialFilter);

	// 	material->setMat4("model", transform->getModelMatrix());

	// 	material->bind(state);

	// 	Mesh* mesh = manager->getMesh(meshRenderers[j]->meshFilter);

	// 	meshVAO[meshRenderers[j]->meshFilter].bind();
	// 	meshVAO[meshRenderers[j]->meshFilter].draw((int)mesh->vertices.size());
	// 	meshVAO[meshRenderers[j]->meshFilter].unbind();
	// }

	// std::vector<Cloth*> cloths = manager->getCloths();
	// for(unsigned int j = 0; j < cloths.size(); j++){
	// 	Transform *transform = cloths[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
	// 	Material *material = manager->getMaterial(2);  // just geeting first material for right now, change later

	// 	material->setMat4("model", transform->getModelMatrix());

	// 	material->bind(state);

	// 	int size = 9*2*(cloths[j]->nx - 1)*(cloths[j]->ny - 1);

	// 	cloths[j]->clothVAO.bind();
	// 	cloths[j]->clothVAO.draw(size);
	// 	cloths[j]->clothVAO.unbind();
	// }

	// std::vector<Solid*> solids = manager->getSolids();
	// for(unsigned int j = 0; j < solids.size(); j++){
	// 	Transform *transform = solids[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
	// 	Material *material = manager->getMaterial(3);  // just geeting first material for right now, change later

	// 	material->setMat4("model", transform->getModelMatrix());

	// 	material->bind(state);

	// 	int size = 3*(solids[j]->ne_b)*(solids[j]->npe_b);//9*2*(cloths[j]->nx - 1)*(cloths[j]->ny - 1);

	// 	solids[j]->solidVAO.bind();
	// 	solids[j]->solidVAO.draw(size);
	// 	solids[j]->solidVAO.unbind();
	// }
}

void RenderSystem::createShadowMaps()
{
	//shadowFBO = new Framebuffer(SHADOW_WIDTH, SHADOW_WIDTH);

	//shadowFBO->generate();

	if (manager->getNumberOfDirectionalLights() > 0){
		for (int i = 0; i < NUM_CASCADES; i++){
			cascadeTexture2D[i] = new Texture2D(SHADOW_WIDTH, SHADOW_HEIGHT, TextureFormat::Depth);

			Graphics::generate(cascadeTexture2D[i]);
		}
	}

	if (manager->getNumberOfSpotLights() > 0){
		shadowTexture2D = new Texture2D(SHADOW_WIDTH, SHADOW_HEIGHT, TextureFormat::Depth);

		Graphics::generate(shadowTexture2D);
	}

	if (manager->getNumberOfPointLights() > 0){
		shadowCubemap = new Cubemap(SHADOW_WIDTH, TextureFormat::Depth);

		Graphics::generate(shadowCubemap);
	}
}

void RenderSystem::calcCascadeOrthoProj(glm::vec3 lightDirection)
{
	glm::mat4 viewInv = glm::inverse(view);

	glm::vec3 direction = -lightDirection;

	float factor = glm::tan(glm::radians(45.0f));

	float cascadeEnds[NUM_CASCADES + 1] = { -0.1f, -2.0f, -4.0f, -10.0f, -20.0f, -100.0f };
	for (unsigned int i = 0; i < NUM_CASCADES; i++){
		float xn = cascadeEnds[i] * factor;
		float xf = cascadeEnds[i + 1] * factor;
		float yn = cascadeEnds[i] * factor;
		float yf = cascadeEnds[i + 1] * factor;

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
		glm::vec4 cW = viewInv * frustumCentre;
		float d = cascadeEnds[i + 1] - cascadeEnds[i];

		glm::vec3 p = glm::vec3(cW.x + d*direction.x, cW.y + d*direction.y, cW.z + d*direction.z);

		cascadeLightView[i] = glm::lookAt(p, glm::vec3(cW.x, cW.y, cW.z), glm::vec3(0.0f, 1.0f, 0.0f));

		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::min();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::min();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::min();
		for (unsigned int j = 0; j < 8; j++){
			// Transform the frustum coordinate from view to world space
			glm::vec4 vW = viewInv * frustumCorners[j];

			// Transform the frustum coordinate from world to light space
			glm::vec4 vL = cascadeLightView[i] * vW;

			minX = glm::min(minX, vL.x);
			maxX = glm::max(maxX, vL.x);
			minY = glm::min(minY, vL.y);
			maxY = glm::max(maxY, vL.y);
			minZ = glm::min(minZ, vL.z);
			maxZ = glm::max(maxZ, vL.z);
		}

		cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, maxZ-minZ);
	}
}

void RenderSystem::calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection)
{
	cubeViewMatrices[0] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[1] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[2] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
	cubeViewMatrices[3] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
	cubeViewMatrices[4] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
	cubeViewMatrices[5] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
}

void RenderSystem::renderShadowMap(Texture2D* texture, glm::mat4 lightView, glm::mat4 lightProjection)
{
	// glDepthFunc(GL_LEQUAL);

	// shadowFBO->bind();
	// shadowFBO->addAttachment2D(texture->getHandle(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0);

	// shadowFBO->clearDepthBuffer(1.0f);

	// depthShader.bind();
	// depthShader.setMat4("view", lightView);
	// depthShader.setMat4("projection", lightProjection);

	// std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
	// for (unsigned int i = 0; i < meshRenderers.size(); i++){
	// 	Transform* transform = meshRenderers[i]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
	// 	depthShader.setMat4("model", transform->getModelMatrix());

	// 	Mesh* mesh = manager->getMesh(meshRenderers[i]->meshFilter);

	// 	meshVAO[meshRenderers[i]->meshFilter].bind();
	// 	meshVAO[meshRenderers[i]->meshFilter].draw((int)mesh->vertices.size());
	// 	meshVAO[meshRenderers[i]->meshFilter].unbind();
	// }

	// shadowFBO->unbind();
}

// void RenderSystem::renderDepthCubemap(Cubemap* cubemap, glm::mat4 lightProjection)
// {
// 	glDepthFunc(GL_LEQUAL);
	
// 	shadowFBO->bind();
// 	for (unsigned int i = 0; i < 6; i++){
// 		shadowFBO->addAttachment2D(cubemap->getHandle(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0);
	
// 		shadowFBO->clearDepthBuffer(1.0f);

// 		depthShader.bind();
// 		depthShader.setMat4("view", cubeViewMatrices[i]);
// 		depthShader.setMat4("projection", lightProjection);

// 		std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
// 		for (unsigned int j = 0; j < meshRenderers.size(); j++){
// 			Transform* transform = meshRenderers[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 			depthShader.setMat4("model", transform->getModelMatrix());

// 			Mesh* mesh = manager->getMesh(meshRenderers[j]->meshFilter);

// 			meshVAO[meshRenderers[j]->meshFilter].bind();
// 			meshVAO[meshRenderers[j]->meshFilter].draw((int)mesh->vertices.size());
// 			meshVAO[meshRenderers[j]->meshFilter].unbind();
// 		}
// 	}

// 	if (Input::getKeyDown(KeyCode::Space)){
// 		cubemap->readPixels();

// 		for (unsigned int i = 0; i < 6; i++){
// 			int width = cubemap->getWidth();
// 			Texture2D face(width, width, TextureFormat::Depth);
// 			face.setPixels(cubemap->getPixels((CubemapFace)(CubemapFace::PositiveX + i)));

// 			std::vector<unsigned char> data = face.getRawTextureData();
// 			std::cout << "face width: " << face.getWidth() << " size of raw texture data: " << data.size() << std::endl;
// 			const std::string name = "cubemap_test" + std::to_string(i) + ".bmp";
// 			TextureLoader::writeToBMP(name, data, SHADOW_WIDTH, SHADOW_WIDTH, 1);
// 		}
// 	}

// 	shadowFBO->unbind();
// }
















































// #include "RenderSystem.h"

// #include <cstddef>

// #include "../graphics/OpenGL.h"
// #include "../graphics/DebugLine.h"

// #include "../core/Debug.h"
// #include "../core/Input.h"
// #include "../core/Log.h"
// #include "../TextureLoader.h"


// using namespace PhysicsEngine;

// const unsigned int SHADOW_WIDTH = 1000, SHADOW_HEIGHT = 1000;
// const unsigned int DEBUG_WIDTH = 1000, DEBUG_HEIGHT = 1000;

// const unsigned int NUM_CASCADES = 5;

// RenderSystem::RenderSystem(Manager *manager)
// {
// 	this->manager = manager;

// 	numLights = 0;

// 	cascadeLightView.resize(NUM_CASCADES);
// 	cascadeOrthoProj.resize(NUM_CASCADES);

// 	cubeViewMatrices.resize(6);

// 	cascadeTexture2D.resize(6);
// 	shadowTexture2D = NULL;
// 	shadowCubemap = NULL;
// 	shadowFBO = NULL;
// }

// RenderSystem::~RenderSystem()
// {
// 	for (unsigned int i = 0; i < NUM_CASCADES; i++){
// 		delete cascadeTexture2D[i];
// 	}

// 	delete shadowTexture2D;
// 	delete shadowCubemap;
// 	delete shadowFBO;
// }


// void RenderSystem::init()
// {
// 	std::vector<Texture2D>& textures = manager->getTextures();
// 	for (unsigned int i = 0; i < textures.size(); i++){
// 		textures[i].generate();
// 	}

// 	std::vector<Shader>& shaders = manager->getShaders();
// 	for (unsigned int i = 0; i < shaders.size(); i++){
// 		if (!shaders[i].compile()){
// 			std::cout << "RenderSystem: shader failed to compile" << std::endl;
// 		}

// 		shaders[i].setUniformBlock("CameraBlock", (int)UniformBuffer::CameraBuffer);
// 		shaders[i].setUniformBlock("ShadowBlock", (int)UniformBuffer::ShadowBuffer);
// 		shaders[i].setUniformBlock("DirectionalLightBlock", (int)UniformBuffer::DirectionalLightBuffer);
// 		shaders[i].setUniformBlock("SpotLightBlock", (int)UniformBuffer::SpotLightBuffer);
// 		shaders[i].setUniformBlock("PointLightBlock", (int)UniformBuffer::PointLightBuffer);
// 	}

// 	if (!depthShader.compile("../data/shaders/depth.vs", "../data/shaders/depth.frag")){
// 		Log::Info("RenderSystem: depth shader failed to compile");
// 	}

// 	if(!particleShader.compile("../data/shaders/particle_directional.vs", "../data/shaders/particle_directional.frag")){
// 		Log::Info("RenderSystem: particle shader failed to compile");
// 	}

// 	particleShader.setUniformBlock("CameraBlock", (int)UniformBuffer::CameraBuffer);

// 	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
// 	std::vector<Mesh> meshes = manager->getMeshes();
// 	meshVAO.resize(meshes.size());
// 	vertexVBO.resize(meshes.size());
// 	normalVBO.resize(meshes.size());
// 	texCoordVBO.resize(meshes.size());
// 	for (unsigned int i = 0; i < meshes.size(); i++){
// 		Mesh* mesh = &meshes[i];

// 		meshVAO[i].generate();
// 		meshVAO[i].bind();

// 		vertexVBO[i].generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 		vertexVBO[i].bind();
// 		vertexVBO[i].setData(&(mesh->vertices[0]), mesh->vertices.size()*sizeof(float));
// 		meshVAO[i].setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 		normalVBO[i].generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 		normalVBO[i].bind();
// 		normalVBO[i].setData(&(mesh->normals[0]), mesh->normals.size()*sizeof(float));
// 		meshVAO[i].setLayout(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 		texCoordVBO[i].generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 		texCoordVBO[i].bind();
// 		texCoordVBO[i].setData(&(mesh->texCoords[0]), mesh->texCoords.size()*sizeof(float));
// 		meshVAO[i].setLayout(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

// 		/*colourVBO[i].generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 		colourVBO[i].bind();
// 		colourVBO[i].setData(&(mesh->getColours()[0]), mesh->getColours().size()*sizeof(float));
// 		meshVAO[i].setLayout(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);*/

// 		meshVAO[i].unbind();
// 	}

// 	state.init();

// 	OpenGL::enableDepthTest();
// 	OpenGL::enableCubemaps();
// 	OpenGL::enablePoints();

// 	createShadowMaps();

// 	OpenGL::checkError();
// }

// void RenderSystem::update()
// {
// 	std::vector<DirectionalLight*> directionalLights = manager->getDirectionalLights();
// 	std::vector<SpotLight*> spotLights = manager->getSpotLights();
// 	std::vector<PointLight*> pointLights = manager->getPointLights();

// 	Camera* camera = manager->getCamera();

// 	OpenGL::setViewport(camera->x, camera->y, camera->width, camera->height);
// 	OpenGL::clearColorBuffer(camera->getBackgroundColor());
// 	OpenGL::clearDepthBuffer(1.0f);

// 	projection = camera->getProjMatrix();
// 	view = camera->getViewMatrix();
// 	cameraPos = camera->getPosition();

// 	state.bind(UniformBuffer::CameraBuffer);
// 	state.setProjectionMatrix(projection);
// 	state.setViewMatrix(view);
// 	state.setCameraPosition(cameraPos);
// 	state.unbind(UniformBuffer::CameraBuffer);

// 	pass = 0;

// 	if (directionalLights.size() >= 1){
// 		glm::vec3 lightDirection = directionalLights[0]->direction;

// 		calcCascadeOrthoProj(lightDirection);

// 		for (unsigned int i = 0; i < NUM_CASCADES; i++){
// 			renderShadowMap(cascadeTexture2D[i], cascadeLightView[i], cascadeOrthoProj[i]);
// 		}

// 		float ends[NUM_CASCADES] = { -2.0f, -4.0f, -10.0f, -20.0f, -100.0f };

// 		state.bind(UniformBuffer::ShadowBuffer);
// 		for (unsigned int i = 0; i < 5; i++){
// 			state.setLightProjectionMatrix(cascadeOrthoProj[i], i);
// 			state.setLightViewMatrix(cascadeLightView[i], i);

// 			glm::vec4 v(0.0f, 0.0f, ends[i], 1.0f);
// 			glm::vec4 zClip = projection*v;

// 			state.setCascadeEnd(zClip.z, i);
// 		}
// 		state.unbind(UniformBuffer::ShadowBuffer);

// 		state.bind(UniformBuffer::DirectionalLightBuffer);
// 		state.setDirLightDirection(-directionalLights[0]->direction);
// 		state.setDirLightAmbient(directionalLights[0]->ambient);
// 		state.setDirLightDiffuse(directionalLights[0]->diffuse);
// 		state.setDirLightSpecular(directionalLights[0]->specular);
// 		state.unbind(UniformBuffer::DirectionalLightBuffer);

// 		state.cascadeTexture2D = cascadeTexture2D;

// 		renderScene();

// 		pass++;
// 	}

// 	for (unsigned int i = 0; i < spotLights.size(); i++){
// 		glm::vec3 position = spotLights[i]->position;
// 		glm::vec3 direction = spotLights[i]->direction;
// 		glm::mat4 lightProjection = spotLights[i]->projection;
// 		glm::mat4 lightView = glm::lookAt(position, position - direction, glm::vec3(0.0f, 1.0f, 0.0f));

// 		renderShadowMap(shadowTexture2D, lightView, lightProjection);

// 		state.bind(UniformBuffer::ShadowBuffer);
// 		state.setLightProjectionMatrix(spotLights[i]->projection, 0);
// 		state.setLightViewMatrix(lightView, 0);
// 		state.unbind(UniformBuffer::ShadowBuffer);

// 		state.bind(UniformBuffer::SpotLightBuffer);
// 		state.setSpotLightPosition(spotLights[i]->position);
// 		state.setSpotLightDirection(spotLights[i]->direction);
// 		state.setSpotLightAmbient(spotLights[i]->ambient);
// 		state.setSpotLightDiffuse(spotLights[i]->diffuse);
// 		state.setSpotLightSpecular(spotLights[i]->specular);
// 		state.setSpotLightConstant(spotLights[i]->constant);
// 		state.setSpotLightLinear(spotLights[i]->linear);
// 		state.setSpotLightQuadratic(spotLights[i]->quadratic);
// 		state.setSpotLightCutoff(spotLights[i]->cutOff);
// 		state.setSpotLightOuterCutoff(spotLights[i]->outerCutOff);
// 		state.unbind(UniformBuffer::SpotLightBuffer);

// 		state.shadowTexture2D = shadowTexture2D;

// 		renderScene();

// 		pass++;
// 	}

// 	for (unsigned int i = 0; i < pointLights.size(); i++){

// 		if (Input::getKey(KeyCode::A)){
// 			pointLights[0]->position.x += 0.1f;
// 		}
// 		if (Input::getKey(KeyCode::S)){
// 			pointLights[0]->position.x -= 0.1f;
// 		}
// 		if (Input::getKey(KeyCode::W)){
// 			pointLights[0]->position.z += 0.1f;
// 		}
// 		if (Input::getKey(KeyCode::Z)){
// 			pointLights[0]->position.z -= 0.1f;
// 		}

// 		glm::vec3 lightPosition = pointLights[i]->position;
// 		glm::mat4 lightProjection = pointLights[i]->projection;

// 		calcCubeViewMatrices(lightPosition, lightProjection);

// 		renderDepthCubemap(shadowCubemap, lightProjection);

// 		state.bind(UniformBuffer::ShadowBuffer);
// 		state.setFarPlane(25.0f);
// 		state.unbind(UniformBuffer::ShadowBuffer);

// 		state.bind(UniformBuffer::PointLightBuffer);
// 		state.setPointLightPosition(pointLights[i]->position);
// 		state.setPointLightAmbient(pointLights[i]->ambient);
// 		state.setPointLightDiffuse(pointLights[i]->diffuse);
// 		state.setPointLightSpecular(pointLights[i]->specular);
// 		state.setPointLightConstant(pointLights[i]->constant);
// 		state.setPointLightLinear(pointLights[i]->linear);
// 		state.setPointLightQuadratic(pointLights[i]->quadratic);
// 		state.unbind(UniformBuffer::PointLightBuffer);

// 		state.shadowCubemap = shadowCubemap;

// 		renderScene();
		
// 		pass++;
// 	}

// 	OpenGL::checkError();
// }

// void RenderSystem::renderScene()
// {
// 	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

// 	std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
// 	for (unsigned int j = 0; j < meshRenderers.size(); j++){
// 		Transform *transform = meshRenderers[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 		Material *material = manager->getMaterial(meshRenderers[j]->materialFilter);

// 		material->setMat4("model", transform->getModelMatrix());

// 		material->bind(state);

// 		Mesh* mesh = manager->getMesh(meshRenderers[j]->meshFilter);

// 		meshVAO[meshRenderers[j]->meshFilter].bind();
// 		meshVAO[meshRenderers[j]->meshFilter].draw((int)mesh->vertices.size());
// 		meshVAO[meshRenderers[j]->meshFilter].unbind();
// 	}

// 	// std::vector<LineRenderer*> lineRenderers = manager->getLineRenderers();
// 	// for (unsigned int j = 0; j < lineRenderers.size(); j++){
// 	// 	Material* material = manager->getMaterial(lineRenderers[j]->materialFilter);

// 	// 	material->bind(state);

// 	// 	material->setMat4("model", glm::mat4(1.0));

// 	// 	lineRenderers[j]->draw();
// 	// }

// 	std::vector<Cloth*> cloths = manager->getCloths();
// 	for(unsigned int j = 0; j < cloths.size(); j++){
// 		Transform *transform = cloths[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 		Material *material = manager->getMaterial(2);  // just geeting first material for right now, change later

// 		material->setMat4("model", transform->getModelMatrix());

// 		material->bind(state);

// 		int size = 9*2*(cloths[j]->nx - 1)*(cloths[j]->ny - 1);

// 		cloths[j]->clothVAO.bind();
// 		cloths[j]->clothVAO.draw(size);
// 		cloths[j]->clothVAO.unbind();
// 	}

// 	std::vector<Solid*> solids = manager->getSolids();
// 	for(unsigned int j = 0; j < solids.size(); j++){
// 		Transform *transform = solids[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 		Material *material = manager->getMaterial(3);  // just geeting first material for right now, change later

// 		material->setMat4("model", transform->getModelMatrix());

// 		material->bind(state);

// 		int size = 3*(solids[j]->ne_b)*(solids[j]->npe_b);//9*2*(cloths[j]->nx - 1)*(cloths[j]->ny - 1);

// 		solids[j]->solidVAO.bind();
// 		solids[j]->solidVAO.draw(size);
// 		solids[j]->solidVAO.unbind();
// 	}

// 	// // maybe temporary?
// 	// particleShader.bind();
// 	// particleShader.setMat4("view", state.getViewMatrix());
// 	// particleShader.setMat4("projection", state.getProjectionMatrix());

// 	// std::vector<Cloth*> cloths = manager->getCloths();
// 	// for(unsigned int i = 0; i < cloths.size(); i++){
// 	// 	Transform *transform = cloths[i]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 	// 	particleShader.setMat4("model", transform->getModelMatrix());

// 	// 	std::vector<float> particles = cloths[i]->particles;
// 	// 	int size = particles.size();

// 	// 	cloths[i]->vao.bind();
// 	// 	cloths[i]->vao.draw(size);
// 	// 	cloths[i]->vao.unbind();
// 	// }

// 	// move to debug system? Probably use input there to toggle it on and off
// 	std::vector<SphereCollider*> sphereColliders = manager->getSphereColliders();
// 	for (unsigned int i = 0; i < sphereColliders.size(); i++){

// 	}
// }

// void RenderSystem::createShadowMaps()
// {
// 	std::vector<DirectionalLight*> directionalLights = manager->getDirectionalLights();
// 	std::vector<SpotLight*> spotLights = manager->getSpotLights();
// 	std::vector<PointLight*> pointLights = manager->getPointLights();

// 	shadowFBO = new Framebuffer(SHADOW_WIDTH, SHADOW_WIDTH);

// 	shadowFBO->generate();

// 	if (directionalLights.size() >= 1){
// 		for (int i = 0; i < NUM_CASCADES; i++){
// 			cascadeTexture2D[i] = new Texture2D(SHADOW_WIDTH, SHADOW_HEIGHT, TextureFormat::Depth);

// 			cascadeTexture2D[i]->generate();
// 		}
// 	}

// 	if (spotLights.size() != 0){
// 		shadowTexture2D = new Texture2D(SHADOW_WIDTH, SHADOW_HEIGHT, TextureFormat::Depth);

// 		shadowTexture2D->generate();
// 	}

// 	if (pointLights.size() != 0){
// 		shadowCubemap = new Cubemap(SHADOW_WIDTH, TextureFormat::Depth);

// 		shadowCubemap->generate();
// 	}
// }

// void RenderSystem::calcCascadeOrthoProj(glm::vec3 lightDirection)
// {
// 	glm::mat4 viewInv = glm::inverse(view);

// 	glm::vec3 direction = -lightDirection;

// 	float factor = glm::tan(glm::radians(45.0f));

// 	float cascadeEnds[NUM_CASCADES + 1] = { -0.1f, -2.0f, -4.0f, -10.0f, -20.0f, -100.0f };
// 	for (unsigned int i = 0; i < NUM_CASCADES; i++){
// 		float xn = cascadeEnds[i] * factor;
// 		float xf = cascadeEnds[i + 1] * factor;
// 		float yn = cascadeEnds[i] * factor;
// 		float yf = cascadeEnds[i + 1] * factor;

// 		glm::vec4 frustumCorners[8];
// 		frustumCorners[0] = glm::vec4(xn, yn, cascadeEnds[i], 1.0f);
// 		frustumCorners[1] = glm::vec4(-xn, yn, cascadeEnds[i], 1.0f);
// 		frustumCorners[2] = glm::vec4(xn, -yn, cascadeEnds[i], 1.0f);
// 		frustumCorners[3] = glm::vec4(-xn, -yn, cascadeEnds[i], 1.0f);

// 		frustumCorners[4] = glm::vec4(xf, yf, cascadeEnds[i + 1], 1.0f);
// 		frustumCorners[5] = glm::vec4(-xf, yf, cascadeEnds[i + 1], 1.0f);
// 		frustumCorners[6] = glm::vec4(xf, -yf, cascadeEnds[i + 1], 1.0f);
// 		frustumCorners[7] = glm::vec4(-xf, -yf, cascadeEnds[i + 1], 1.0f);

// 		glm::vec4 frustumCentre = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

// 		for (int j = 0; j < 8; j++){
// 			frustumCentre.x += frustumCorners[j].x;
// 			frustumCentre.y += frustumCorners[j].y;
// 			frustumCentre.z += frustumCorners[j].z;
// 		}

// 		frustumCentre.x = frustumCentre.x / 8;
// 		frustumCentre.y = frustumCentre.y / 8;
// 		frustumCentre.z = frustumCentre.z / 8;

// 		// Transform the frustum centre from view to world space
// 		glm::vec4 cW = viewInv * frustumCentre;
// 		float d = cascadeEnds[i + 1] - cascadeEnds[i];

// 		glm::vec3 p = glm::vec3(cW.x + d*direction.x, cW.y + d*direction.y, cW.z + d*direction.z);

// 		cascadeLightView[i] = glm::lookAt(p, glm::vec3(cW.x, cW.y, cW.z), glm::vec3(0.0f, 1.0f, 0.0f));

// 		float minX = std::numeric_limits<float>::max();
// 		float maxX = std::numeric_limits<float>::min();
// 		float minY = std::numeric_limits<float>::max();
// 		float maxY = std::numeric_limits<float>::min();
// 		float minZ = std::numeric_limits<float>::max();
// 		float maxZ = std::numeric_limits<float>::min();
// 		for (unsigned int j = 0; j < 8; j++){
// 			// Transform the frustum coordinate from view to world space
// 			glm::vec4 vW = viewInv * frustumCorners[j];

// 			// Transform the frustum coordinate from world to light space
// 			glm::vec4 vL = cascadeLightView[i] * vW;

// 			minX = glm::min(minX, vL.x);
// 			maxX = glm::max(maxX, vL.x);
// 			minY = glm::min(minY, vL.y);
// 			maxY = glm::max(maxY, vL.y);
// 			minZ = glm::min(minZ, vL.z);
// 			maxZ = glm::max(maxZ, vL.z);
// 		}

// 		cascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, maxZ-minZ);
// 	}
// }

// void RenderSystem::calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection)
// {
// 	/*cubeViewMatrices[0] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[1] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[2] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
// 	cubeViewMatrices[3] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
// 	cubeViewMatrices[4] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[5] = (lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));*/
// 	cubeViewMatrices[0] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[1] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[2] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
// 	cubeViewMatrices[3] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
// 	cubeViewMatrices[4] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
// 	cubeViewMatrices[5] = (glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
// }

// void RenderSystem::renderShadowMap(Texture2D* texture, glm::mat4 lightView, glm::mat4 lightProjection)
// {
// 	glDepthFunc(GL_LEQUAL);

// 	shadowFBO->bind();
// 	shadowFBO->addAttachment2D(texture->getHandle(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0);

// 	shadowFBO->clearDepthBuffer(1.0f);

// 	depthShader.bind();
// 	depthShader.setMat4("view", lightView);
// 	depthShader.setMat4("projection", lightProjection);

// 	std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
// 	for (unsigned int i = 0; i < meshRenderers.size(); i++){
// 		Transform* transform = meshRenderers[i]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 		depthShader.setMat4("model", transform->getModelMatrix());

// 		Mesh* mesh = manager->getMesh(meshRenderers[i]->meshFilter);

// 		meshVAO[meshRenderers[i]->meshFilter].bind();
// 		meshVAO[meshRenderers[i]->meshFilter].draw((int)mesh->vertices.size());
// 		meshVAO[meshRenderers[i]->meshFilter].unbind();
// 	}

// 	shadowFBO->unbind();
// }

// void RenderSystem::renderDepthCubemap(Cubemap* cubemap, glm::mat4 lightProjection)
// {
// 	glDepthFunc(GL_LEQUAL);
	
// 	shadowFBO->bind();
// 	for (unsigned int i = 0; i < 6; i++){
// 		shadowFBO->addAttachment2D(cubemap->getHandle(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0);
	
// 		shadowFBO->clearDepthBuffer(1.0f);

// 		depthShader.bind();
// 		depthShader.setMat4("view", cubeViewMatrices[i]);
// 		depthShader.setMat4("projection", lightProjection);

// 		std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();
// 		for (unsigned int j = 0; j < meshRenderers.size(); j++){
// 			Transform* transform = meshRenderers[j]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
// 			depthShader.setMat4("model", transform->getModelMatrix());

// 			Mesh* mesh = manager->getMesh(meshRenderers[j]->meshFilter);

// 			meshVAO[meshRenderers[j]->meshFilter].bind();
// 			meshVAO[meshRenderers[j]->meshFilter].draw((int)mesh->vertices.size());
// 			meshVAO[meshRenderers[j]->meshFilter].unbind();
// 		}
// 	}

// 	if (Input::getKeyDown(KeyCode::Space)){
// 		cubemap->readPixels();

// 		for (unsigned int i = 0; i < 6; i++){
// 			int width = cubemap->getWidth();
// 			Texture2D face(width, width, TextureFormat::Depth);
// 			face.setPixels(cubemap->getPixels((CubemapFace)(CubemapFace::PositiveX + i)));

// 			std::vector<unsigned char> data = face.getRawTextureData();
// 			std::cout << "face width: " << face.getWidth() << " size of raw texture data: " << data.size() << std::endl;
// 			const std::string name = "cubemap_test" + std::to_string(i) + ".bmp";
// 			TextureLoader::writeToBMP(name, data, SHADOW_WIDTH, SHADOW_WIDTH, 1);
// 		}
// 	}

// 	shadowFBO->unbind();
// }