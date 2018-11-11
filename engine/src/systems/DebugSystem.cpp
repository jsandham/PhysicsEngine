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
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * vec4(position, 1.0);\n"
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

	// Gizmos::init();
}

void DebugSystem::update()
{
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




	Camera* camera;
	if(manager->getNumberOfComponents<Camera>() > 0){
		camera = manager->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	if (Input::getKeyDown(KeyCode::I)){
		Entity* entity = manager->instantiate();
		if(entity != NULL){
			Transform* transform = entity->addComponent<Transform>();
			LineRenderer* lineRenderer = entity->addComponent<LineRenderer>();

			lineRenderer->materialId = lineMaterial->assetId;

			glm::mat4 view = camera->getViewMatrix();
			glm::mat4 projection = camera->getProjMatrix();
			glm::mat4 projViewInv = glm::inverse(projection * view);
			//glm::mat4 inverseView = glm::inverse(view);
			//glm::mat4 inverseProj = glm::inverse(projection);

			// std::cout << "view: " << std::endl;
			// std::cout << view[0][0] << " " << view[0][1] << " " << view[0][2] << " " << view[0][3] << std::endl;
			// std::cout << view[1][0] << " " << view[1][1] << " " << view[1][2] << " " << view[1][3] << std::endl;
			// std::cout << view[2][0] << " " << view[2][1] << " " << view[2][2] << " " << view[2][3] << std::endl;
			// std::cout << view[3][0] << " " << view[3][1] << " " << view[3][2] << " " << view[3][3] << std::endl;

			// std::cout << "inverse: " << std::endl;
			// std::cout << inverse[0][0] << " " << inverse[0][1] << " " << inverse[0][2] << " " << inverse[0][3] << std::endl;
			// std::cout << inverse[1][0] << " " << inverse[1][1] << " " << inverse[1][2] << " " << inverse[1][3] << std::endl;
			// std::cout << inverse[2][0] << " " << inverse[2][1] << " " << inverse[2][2] << " " << inverse[2][3] << std::endl;
			// std::cout << inverse[3][0] << " " << inverse[3][1] << " " << inverse[3][2] << " " << inverse[3][3] << std::endl;

			// std::cout << "projection: " << std::endl;
			// std::cout << projection[0][0] << " " << projection[0][1] << " " << projection[0][2] << " " << projection[0][3] << std::endl;
			// std::cout << projection[1][0] << " " << projection[1][1] << " " << projection[1][2] << " " << projection[1][3] << std::endl;
			// std::cout << projection[2][0] << " " << projection[2][1] << " " << projection[2][2] << " " << projection[2][3] << std::endl;
			// std::cout << projection[3][0] << " " << projection[3][1] << " " << projection[3][2] << " " << projection[3][3] << std::endl;

			// std::cout << "inverse proj: " << std::endl;
			// std::cout << inverseProj[0][0] << " " << inverseProj[0][1] << " " << inverseProj[0][2] << " " << inverseProj[0][3] << std::endl;
			// std::cout << inverseProj[1][0] << " " << inverseProj[1][1] << " " << inverseProj[1][2] << " " << inverseProj[1][3] << std::endl;
			// std::cout << inverseProj[2][0] << " " << inverseProj[2][1] << " " << inverseProj[2][2] << " " << inverseProj[2][3] << std::endl;
			// std::cout << inverseProj[3][0] << " " << inverseProj[3][1] << " " << inverseProj[3][2] << " " << inverseProj[3][3] << std::endl;

			std::cout << "projViewInv: " << std::endl;
			std::cout << projViewInv[0][0] << " " << projViewInv[0][1] << " " << projViewInv[0][2] << " " << projViewInv[0][3] << std::endl;
			std::cout << projViewInv[1][0] << " " << projViewInv[1][1] << " " << projViewInv[1][2] << " " << projViewInv[1][3] << std::endl;
			std::cout << projViewInv[2][0] << " " << projViewInv[2][1] << " " << projViewInv[2][2] << " " << projViewInv[2][3] << std::endl;
			std::cout << projViewInv[3][0] << " " << projViewInv[3][1] << " " << projViewInv[3][2] << " " << projViewInv[3][3] << std::endl;

			int x = Input::getMousePosX();
			int y = Input::getMousePosY();
			int width = camera->width;
			int height = camera->height;

			std::cout << "x: " << x << " y: " << y << " width: " << width << " height: " << height << std::endl;

			float screen_x = (x - 0.5f * width) / (0.5f * width);
			float screen_y = (0.5f * height - y) / (0.5f * height);

			std::cout << "screen_x: " << screen_x << " screen_y: " << screen_y << std::endl;

			glm::vec4 nearPoint = projViewInv * glm::vec4(screen_x, screen_y, 0, 1);
			glm::vec4 farPoint = projViewInv * glm::vec4(screen_x, screen_y, 1, 1);

			std::cout << "near point: " << nearPoint.x << " " << nearPoint.y << " " << nearPoint.z << "far point: " << farPoint.x << " " << farPoint.y << " " << farPoint.z << std::endl;

			lineRenderer->start = glm::vec3(nearPoint.x / nearPoint.w, nearPoint.y / nearPoint.w, nearPoint.z / nearPoint.w);
			lineRenderer->end = glm::vec3(farPoint.x / farPoint.w, farPoint.y / farPoint.w, farPoint.z / farPoint.w);



			// std::vector<float> vertices;
			// vertices.resize(6);
			// vertices[0] = p1.x;
			// vertices[1] = p1.y;
			// vertices[2] = p1.z;
			// vertices[3] = p2.x;
			// vertices[4] = p2.y;
			// vertices[5] = p2.z;

			// std::cout << "start: " << p1.x << " " << p1.y << " " << p1.z << " end: " << p2.x << " " << p2.y << " " << p2.z << std::endl;

			// //std::vector<float> vertices = { 0.2f, 0.0f, -1.0f, 0.0f, 1.0f, -10.0f };
			// lineRenderer->setVertices(vertices);
			// lineRenderer->setMaterialFilter(manager->getMaterialFilter("lineMat"));
			// lineRenderer->initLineData();
			// lineRenderer->updateLineData();
		}

		//entity->addComponent<Transform>(transform);
		//entity->addComponent<LineRenderer>(lineRenderer);

		//Debug::drawLine(glm::vec3(0.2,0,-1), glm::vec3(0,1,-10));
		//std::cout << "space bar pressed" << std::endl;
	}
}