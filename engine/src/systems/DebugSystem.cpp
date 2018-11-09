#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/Input.h"
#include "../../include/core/Manager.h"

#include "../../include/components/Transform.h"
#include "../../include/components/LineRenderer.h"

#include "../../include/glm/glm.hpp"
#include "../../include/glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

std::string vertexShader = "#version 330 core\n"
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
"}";

std::string fragmentShader = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

DebugSystem::DebugSystem()
{
	type = 3;
}

DebugSystem::DebugSystem(unsigned char* data)
{
	type = 3;
}

DebugSystem::~DebugSystem()
{

}

void DebugSystem::init()
{
	lineMaterial = manager->create<Material>();
	lineMaterial->setManager(manager);
	lineShader = manager->create<Shader>();

	lineShader->vertexShader = vertexShader;
	lineShader->fragmentShader = fragmentShader;

	lineShader->compile();

	lineMaterial->shaderId = lineShader->assetId;

	/*
	if (!lineMaterial->getShader()->compile()){
		std::cout << "shader failed to compile" << std::endl;
	}*/

	// Gizmos::init();
}

void DebugSystem::update()
{
	// if (Input::getKeyDown(KeyCode::Space)){
	// 	Camera* camera = manager->getCamera();

	// 	Entity* entity = manager->createEntity();
	// 	Transform* transform = manager->createTransform();

	// 	LineRenderer* lineRenderer = manager->createLineRenderer();

	// 	glm::mat4 view = camera->getViewMatrix();

	// 	glm::vec4 start = glm::vec4(0.0f, 0.0f, 0.0f, 1.0);
	// 	glm::vec4 end = glm::vec4(0.0f, 0.0f, 10.0f, 1.0f);

	// 	glm::vec4 p1 = glm::inverse(view)*start;
	// 	glm::vec4 p2 = glm::inverse(view)*end;

	// 	std::vector<float> vertices;
	// 	vertices.resize(6);
	// 	vertices[0] = p1.x;
	// 	vertices[1] = p1.y;
	// 	vertices[2] = p1.z;
	// 	vertices[3] = p2.x;
	// 	vertices[4] = p2.y;
	// 	vertices[5] = p2.z;

	// 	std::cout << "start: " << p1.x << " " << p1.y << " " << p1.z << " end: " << p2.x << " " << p2.y << " " << p2.z << std::endl;

	// 	//std::vector<float> vertices = { 0.2f, 0.0f, -1.0f, 0.0f, 1.0f, -10.0f };
	// 	lineRenderer->setVertices(vertices);
	// 	lineRenderer->setMaterialFilter(manager->getMaterialFilter("lineMat"));
	// 	lineRenderer->initLineData();
	// 	lineRenderer->updateLineData();

	// 	entity->addComponent<Transform>(transform);
	// 	entity->addComponent<LineRenderer>(lineRenderer);

	// 	//Debug::drawLine(glm::vec3(0.2,0,-1), glm::vec3(0,1,-10));
	// 	//std::cout << "space bar pressed" << std::endl;
	// }

	// Camera* camera = manager->getCamera();

	//std::vector<MeshRenderer*> meshRenderers = manager->getMeshRenderers();

	//Mesh *mesh = manager->getMesh(meshRenderers[1]->getMeshFilter());
	//Transform* transform = meshRenderers[1]->entity->getComponent<Transform>(manager);

	// Gizmos::projection = camera->getProjMatrix();
	// Gizmos::view = camera->getViewMatrix();

	//Gizmos::drawWireCube(glm::vec3(1.0, 3.0, 1.0), glm::vec3(0.5, 2.0, 0.5), Color::green);
	// Gizmos::drawWireCube(glm::vec3(2.0, 1.5, 1.0), glm::vec3(0.5, 1.0, 0.5), Color::yellow);
	// Gizmos::drawWireCube(glm::vec3(0.0, 0.5, 1.0), glm::vec3(1.5, 2.0, 1.5), Color::red);
	// Gizmos::drawWireCube(glm::vec3(3.0, 2.0, 1.0), glm::vec3(0.5, 0.2, 0.5), Color::blue);
	// Gizmos::drawWireCube(glm::vec3(4.0, 0.0, 1.0), glm::vec3(0.2, 1.0, 1.0), Color::green);

	//Gizmos::drawWireSphere(glm::vec3(-2.0f, 1.0f, 2.0f), 0.7f, Color::black);
	//Gizmos::drawWireMesh(mesh, glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), Color::red);

	//Octtree tree = Physics::getOcttree();
	//Gizmos::drawOcttree(&tree, Color::yellow);




	//std::cout << "LogicSystem update called" << std::endl;

	//std::cout << "line material guid: " << lineMaterial->assetId.toString() << "shader id: " << lineMaterial->shaderId.toString() << std::endl;

	if(Input::getKeyDown(KeyCode::I)){
		Entity* entity = manager->instantiate();
		if(entity != NULL){
			Transform* transform = entity->addComponent<Transform>();
			LineRenderer* lineRenderer = entity->addComponent<LineRenderer>();
			lineRenderer->materialId = lineMaterial->assetId;
			lineRenderer->end = glm::vec3(10.0f, 0.0f, 1.0f);
			std::cout << "Creating new entity with id: " << entity->entityId.toString() << " total entities now: " << manager->getNumberOfEntities() << " creating line renderer: " << lineRenderer->componentId.toString() << " total number of line renderers: " << manager->getNumberOfComponents<LineRenderer>() << " start: " << lineRenderer->start.x << " " << lineRenderer->start.y << " " << lineRenderer->start.z << " end: " << lineRenderer->end.x << " " << lineRenderer->end.y << " " << lineRenderer->end.z << std::endl;
		}
	}
}