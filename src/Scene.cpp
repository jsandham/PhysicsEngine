#include "Scene.h"
#include "graphics/Material.h"
#include "core/Log.h"
#include "core/Input.h"

using namespace PhysicsEngine;

Scene::Scene()
{

}

Scene::~Scene()
{

}

void Scene::init()
{
	Log::Info("scene init called %d\n", 4);

	// load assets
	manager.loadMesh("../data/meshes/square.txt");
	manager.loadMesh("../data/meshes/cube.txt");
	manager.loadMesh("../data/meshes/square.txt");
	manager.loadMesh("../data/meshes/teapot.txt");
	manager.loadMesh("../data/meshes/dragon.txt");
	manager.loadMesh("../data/meshes/cow.txt");
	manager.loadMesh("../data/meshes/sphere.txt");

	manager.loadTexture2D("../data/textures/default.png");
	manager.loadTexture2D("../data/textures/blue.png");
	manager.loadTexture2D("../data/textures/sun.png");

	manager.loadShader("standard", "../data/shaders/standard_directional.vs", "../data/shaders/standard_directional.frag");
	manager.loadShader("particle", "../data/shaders/particle_directional.vs", "../data/shaders/particle_directional.frag");
	manager.loadShader("line", "../data/shaders/line.vs", "../data/shaders/line.frag");

	std::vector<std::string> faces = {"../data/textures/right.jpg", 
		 							  "../data/textures/left.jpg",
		 							  "../data/textures/top.jpg",
		 							  "../data/textures/bottom.jpg",
		 							  "../data/textures/back.jpg",
		 							  "../data/textures/front.jpg"}; 

	manager.loadCubemap(faces);

	Material *defaultMat = new Material(manager.getShader("standard"));
	defaultMat->setMainTexture(manager.getTexture2D("../data/textures/default.png"));
	Material *sunMat = new Material(manager.getShader("standard"));
	sunMat->setMainTexture(manager.getTexture2D("../data/textures/sun.png"));
	Material *lineMat = new Material(manager.getShader("line"));

	manager.loadMaterial("defaultMat", *defaultMat);
	manager.loadMaterial("sunMat", *sunMat);
	manager.loadMaterial("lineMat", *lineMat);

	delete defaultMat;
	delete sunMat;
	delete lineMat;

	Log::Info("assets loaded");

	// systems
	renderSystem = new RenderSystem(&manager);
	physicsSystem = new PhysicsSystem(&manager);
	cleanUpSystem = new CleanUpSystem(&manager);
	debugSystem = new DebugSystem(&manager);
	playerSystem = new PlayerSystem(&manager);

	// lights
	Entity* entity0 = manager.createEntity();
	Camera* camera = manager.createCamera();
	DirectionalLight* directionalLight1 = manager.createDirectionalLight();

	entity0->addComponent<Camera>(camera);
	entity0->addComponent<DirectionalLight>(directionalLight1);

	// entity1 (ground plane)
	Entity* entity1 = manager.createEntity();
	Transform* transform1 = manager.createTransform();
	MeshRenderer* meshRenderer1 = manager.createMeshRenderer();

	transform1->position = glm::vec3(0.0f, 0.0f, 0.0f);
	transform1->setEulerAngles(glm::vec3(3.14159265 / 2.0f, 0.0f, 0.0f));
	transform1->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	entity1->addComponent<Transform>(transform1);
	entity1->addComponent<MeshRenderer>(meshRenderer1);

	meshRenderer1->setMeshFilter(manager.getMeshFilter("../data/meshes/square.txt"));
	meshRenderer1->setMaterialFilter(manager.getMaterialFilter("defaultMat"));

	// entity2 (cube)
	Entity* entity2 = manager.createEntity();
	Transform* transform2 = manager.createTransform();
	MeshRenderer* meshRenderer2 = manager.createMeshRenderer();
	Rigidbody* rigidbody2 = manager.createRigidbody();
	SpringJoint* springJoint2 = manager.createSpringJoint();

	meshRenderer2->setMeshFilter(manager.getMeshFilter("../data/meshes/cube.txt"));
	meshRenderer2->setMaterialFilter(manager.getMaterialFilter("sunMat"));

	transform2->position = glm::vec3(0.0f, 4.0f, 0.0f);
	transform2->setEulerAngles(glm::vec3(3.14159265/2.0f, 0.0f, 0.0f));
	transform2->scale = glm::vec3(0.5f, 0.5f, 0.5f);

	springJoint2->setConnectedAnchor(glm::vec3(0.0f, 5.0f, 0.0f));
	springJoint2->setAnchor(glm::vec3(0.0f, 4.0f, 0.0f));
	springJoint2->restLength = 1.0f;

	entity2->addComponent<Transform>(transform2);
	entity2->addComponent<MeshRenderer>(meshRenderer2);
	entity2->addComponent<Rigidbody>(rigidbody2);
	entity2->addComponent<SpringJoint>(springJoint2);

	// entity3 (teapot)
	Entity* entity3 = manager.createEntity();
	Transform* transform3 = manager.createTransform();
	MeshRenderer* meshRenderer3 = manager.createMeshRenderer();
	Rigidbody* rigidbody3 = manager.createRigidbody();
	SpringJoint* springJoint3 = manager.createSpringJoint();

	meshRenderer3->setMeshFilter(manager.getMeshFilter("../data/meshes/teapot.txt"));
	meshRenderer3->setMaterialFilter(manager.getMaterialFilter("sunMat"));

	transform3->position = glm::vec3(0.0f, 2.0f, 0.0f);
	transform3->setEulerAngles(glm::vec3(0.0f, 0.0f, 0.0f));
	transform3->scale = glm::vec3(0.05f, 0.05f, 0.05f);

	springJoint3->setConnectedBody(rigidbody2);
	springJoint3->setAnchor(glm::vec3(0.0f, 2.0f, 0.0f));
	springJoint3->restLength = 1.0f;

	entity3->addComponent<Transform>(transform3);
	entity3->addComponent<MeshRenderer>(meshRenderer3);
	entity3->addComponent<Rigidbody>(rigidbody3);
	entity3->addComponent<SpringJoint>(springJoint3);








	// entity4 (collider 1)
	// Entity* entity4 = manager.createEntity();
	// Transform* transform4 = manager.createTransform();
	// BoxCollider* boxCollider4 = manager.createBoxCollider();

	// boxCollider4->bounds = Bounds(glm::vec3(-1.0f, 1.0f, 1.0f), glm::vec3(0.4f, 0.5f, 0.7f));

	// entity4->addComponent<Transform>(transform4);
	// entity4->addComponent<BoxCollider>(boxCollider4);

	// // entity5 (collider 2)
	// Entity* entity5 = manager.createEntity();
	// Transform* transform5 = manager.createTransform();
	// BoxCollider* boxCollider5 = manager.createBoxCollider();

	// boxCollider5->bounds = Bounds(glm::vec3(3.0f, 0.5f, -1.0f), glm::vec3(0.4f, 0.2f, 1.0f));

	// entity5->addComponent<Transform>(transform5);
	// entity5->addComponent<BoxCollider>(boxCollider5);

	// // entity6 (collider 3)
	// Entity* entity6 = manager.createEntity();
	// Transform* transform6 = manager.createTransform();
	// BoxCollider* boxCollider6 = manager.createBoxCollider();

	// boxCollider6->bounds = Bounds(glm::vec3(2.0f, 1.5f, 3.0f), glm::vec3(0.1f, 0.1f, 0.2f));

	// entity6->addComponent<Transform>(transform6);
	// entity6->addComponent<BoxCollider>(boxCollider6);

	// // entity7 (collider 4)
	// Entity* entity7 = manager.createEntity();
	// Transform* transform7 = manager.createTransform();
	// BoxCollider* boxCollider7 = manager.createBoxCollider();

	// boxCollider7->bounds = Bounds(glm::vec3(0.5f, 1.2f, 0.75f), glm::vec3(0.5f, 0.3f, 0.6f));

	// entity7->addComponent<Transform>(transform7);
	// entity7->addComponent<BoxCollider>(boxCollider7);

	renderSystem->init();
	physicsSystem->init();
	playerSystem->init();
	cleanUpSystem->init();
	debugSystem->init();
}

void Scene::update()
{
	renderSystem->update();
	physicsSystem->update();
	playerSystem->update();
	cleanUpSystem->update();
	debugSystem->update();
}