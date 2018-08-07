#include "../../include/core/Scene.h"

using namespace PhysicsEngine;

Scene::Scene()
{
	physicsSystem = NULL;
	renderSystem = NULL;
	playerSystem = NULL;
}

Scene::~Scene()
{
	delete physicsSystem;
	delete renderSystem;
	delete playerSystem;
}

void Scene::load(std::string sceneFilePath, std::vector<std::string> assetFilePaths)
{
	sceneName = sceneFilePath;

	std::cout << "scene: " << sceneFilePath << std::endl;

	for(unsigned int i = 0; i < assetFilePaths.size(); i++){
		std::cout << "asset file: " << assetFilePaths[i] << std::endl;
	}

	manager.load(sceneFilePath, assetFilePaths);

	// systems
	renderSystem = new RenderSystem(&manager);
	physicsSystem = new PhysicsSystem(&manager);
	playerSystem = new PlayerSystem(&manager);
}

void Scene::init()
{
	physicsSystem->init();
	renderSystem->init();
	playerSystem->init();
}

void Scene::update()
{
	physicsSystem->update();
	renderSystem->update();
	playerSystem->update();
}





















// // load assets
	// manager.loadGMesh("../data/gmeshes/mesh12.msh");

	// manager.loadMesh("../data/meshes/square.txt");
	// manager.loadMesh("../data/meshes/cube.txt");
	// manager.loadMesh("../data/meshes/square.txt");
	// manager.loadMesh("../data/meshes/teapot.txt");
	// manager.loadMesh("../data/meshes/dragon.txt");
	// manager.loadMesh("../data/meshes/cow.txt");
	// manager.loadMesh("../data/meshes/sphere.txt");

	// manager.loadTexture2D("../data/textures/default.png");
	// manager.loadTexture2D("../data/textures/blue.png");
	// manager.loadTexture2D("../data/textures/sun.png");

	// manager.loadShader("standard", "../data/shaders/standard_directional.vs", "../data/shaders/standard_directional.frag");
	// manager.loadShader("particle", "../data/shaders/particle_directional.vs", "../data/shaders/particle_directional.frag");
	// manager.loadShader("basic", "../data/shaders/basic.vs", "../data/shaders/basic.frag");
	// manager.loadShader("basic2", "../data/shaders/basic2.vs", "../data/shaders/basic2.frag");
	// manager.loadShader("line", "../data/shaders/line.vs", "../data/shaders/line.frag");

	// std::vector<std::string> faces = {"../data/textures/right.jpg", 
	// 	 							  "../data/textures/left.jpg",
	// 	 							  "../data/textures/top.jpg",
	// 	 							  "../data/textures/bottom.jpg",
	// 	 							  "../data/textures/back.jpg",
	// 	 							  "../data/textures/front.jpg"}; 

	// manager.loadCubemap(faces);

	// Material *defaultMat = new Material(manager.getShader("standard"));
	// defaultMat->setMainTexture(manager.getTexture2D("../data/textures/default.png"));
	// Material *sunMat = new Material(manager.getShader("standard"));
	// sunMat->setMainTexture(manager.getTexture2D("../data/textures/sun.png"));
	// Material *basicMat = new Material(manager.getShader("basic"));
	// Material *basic2Mat = new Material(manager.getShader("basic2"));
	// Material *lineMat = new Material(manager.getShader("line"));

	// manager.loadMaterial("defaultMat", *defaultMat);
	// manager.loadMaterial("sunMat", *sunMat);
	// manager.loadMaterial("basicMat", *basicMat);
	// manager.loadMaterial("basic2Mat", *basic2Mat);
	// manager.loadMaterial("lineMat", *lineMat);

	// delete defaultMat;
	// delete sunMat;
	// delete basicMat;
	// delete basic2Mat;
	// delete lineMat;

	// // create cloth particles
	// std::vector<float> particles;
	// std::vector<int> particleTypes;
	// particleTypes.resize(256*256);
	// Util::gridOfParticlesXZPlane(particles, 0.005f, 0.005f, 1.0f, 256, 256);

	// // grab finite element gmesh
	// GMesh* gmesh = manager.getGMesh("../data/gmeshes/mesh12.msh");

	// // AMG tests
	// //int erro_code;
	// //erro_code = DEBUG_TEST_MATRIX_AMG("../data/matrices/mesh1em6.mtx", 177, 49, 0);
	// //erro_code = DEBUG_TEST_MATRIX_AMG("../data/matrices/mesh2em5.mtx", 1162, 307, 0);
	// //erro_code = DEBUG_TEST_MATRIX_AMG("../data/matrices/mesh3em5.mtx", 1089, 290, 0);

	// Log::Info("assets loaded");
	
	// std::vector<int> connect = gmesh->connect;
	// for(unsigned int i = 0; i < 20; i++){
	// 	std::cout << connect[i] << std::endl;
	// }

	// std::cout << "assets loaded" << std::endl;

	// std::vector<int> bconnect = gmesh->bconnect;
	// for(unsigned int i = 0; i < 20; i++){
	// 	std::cout << bconnect[i] << std::endl;
	// }


	// // systems
	// renderSystem = new RenderSystem(&manager);
	// physicsSystem = new PhysicsSystem(&manager);
	// cleanUpSystem = new CleanUpSystem(&manager);
	// debugSystem = new DebugSystem(&manager);
	// playerSystem = new PlayerSystem(&manager);

	// std::cout << "systems created" << std::endl;

	// // lights
	// Entity* entity0 = manager.createEntity();
	// Camera* camera = manager.createCamera();
	// DirectionalLight* directionalLight1 = manager.createDirectionalLight();

	// entity0->addComponent<Camera>(camera);
	// entity0->addComponent<DirectionalLight>(directionalLight1);

	// // entity1 (ground plane)
	// Entity* entity1 = manager.createEntity();
	// Transform* transform1 = manager.createTransform();
	// MeshRenderer* meshRenderer1 = manager.createMeshRenderer();

	// transform1->position = glm::vec3(0.0f, -8.0f, 0.0f);
	// transform1->setEulerAngles(glm::vec3(3.14159265 / 2.0f, 0.0f, 0.0f));
	// transform1->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	// entity1->addComponent<Transform>(transform1);
	// entity1->addComponent<MeshRenderer>(meshRenderer1);

	// meshRenderer1->meshFilter = manager.getMeshFilter("../data/meshes/square.txt");
	// meshRenderer1->materialFilter = manager.getMaterialFilter("defaultMat");

	// // entity2 (cloth)
	// Entity* entity2 = manager.createEntity();
	// Transform* transform2 = manager.createTransform();
	// Cloth* cloth2 = manager.createCloth();

	// transform2->position = glm::vec3(0.0f, 1.0f, 0.0f);
	// transform2->setEulerAngles(glm::vec3(0.0f, 0.0f, 0.0f));
	// transform2->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	// cloth2->nx = 256;
	// cloth2->ny = 256;
	// cloth2->particles = particles;
	// cloth2->particleTypes = particleTypes;

	// entity2->addComponent<Transform>(transform2);
	// entity2->addComponent<Cloth>(cloth2);

	// // entity2 (fem)
	// Entity* entity3 = manager.createEntity();
	// Transform* transform3 = manager.createTransform();
	// Solid* solid3 = manager.createSolid();

	// transform3->position = glm::vec3(0.0f, 1.0f, 0.0f);
	// transform3->setEulerAngles(glm::vec3(0.0f, 0.0f, 0.0f));
	// transform3->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	// solid3->c = 1.0f;                       
 //    solid3->rho = 1.0f;                              
 //    solid3->Q = 1.0f;        
 //    solid3->k = 1.0f;        

 //    solid3->dim = gmesh->dim;               
 //    solid3->ng = gmesh->ng;             
 //    solid3->n = gmesh->n;                            
 //    solid3->nte = gmesh->nte;                
 //    solid3->ne = gmesh->ne;                        
 //    solid3->ne_b = gmesh->ne_b;                                        
 //    solid3->npe = gmesh->npe;                
 //    solid3->npe_b = gmesh->npe_b;               
 //    solid3->type = gmesh->type;                           
 //    solid3->type_b = gmesh->type_b;            
	
	// solid3->vertices = gmesh->vertices;
	// solid3->connect = gmesh->connect;
	// solid3->bconnect = gmesh->bconnect;
	// solid3->groups = gmesh->groups;

	// entity3->addComponent<Transform>(transform3);
	// entity3->addComponent<Solid>(solid3);

	// // entity3 (sphere)
	// Entity* entity3 = manager.createEntity();
	// Transform* transform3 = manager.createTransform();
	// MeshRenderer* meshRenderer3 = manager.createMeshRenderer();
	// meshRenderer3->setMeshFilter(manager.getMeshFilter("../data/meshes/sphere.txt"));
	// meshRenderer3->setMaterialFilter(manager.getMaterialFilter("sunMat"));

	// transform3->position = glm::vec3(0.6f, 1.5f, 0.5f);
	// transform3->setEulerAngles(glm::vec3(3.14159265/2.0f, 0.0f, 0.0f));
	// transform3->scale = glm::vec3(0.25f, 0.25f, 0.25f);

	// entity3->addComponent<Transform>(transform3);
	// entity3->addComponent<MeshRenderer>(meshRenderer3);

	// std::cout << "scene created" << std::endl;

	// physicsSystem->init();
	// renderSystem->init();
	// playerSystem->init();
	// cleanUpSystem->init();
	// debugSystem->init();