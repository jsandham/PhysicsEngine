#include <iostream>

#include <core/Manager.h>
#include <core/input.h>

#include "../include/systems/LogicSystem.h"

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

LogicSystem::LogicSystem()
{
	type = 10;
}

LogicSystem::LogicSystem(unsigned char* data)
{
	type = 10;

	//lineMaterial = manager->create<Material>();//new Material();
}

LogicSystem::~LogicSystem()
{
	//delete lineMaterial;
}

void LogicSystem::init()
{
	std::cout << "number of materials: " << manager->getNumberOfAssets<Material>() << " number of shaders: " << manager->getNumberOfAssets<Shader>() << std::endl;

	lineMaterial = manager->create<Material>();
	lineShader = manager->create<Shader>();

	lineShader->vertexShader = vertexShader;
	lineShader->fragmentShader = fragmentShader;

	lineMaterial->shaderId = lineShader->assetId;

	std::cout << "line material guid: " << lineMaterial->assetId.toString() << "shader id on material: " << lineMaterial->shaderId.toString() << " shader id: " << lineShader->assetId.toString() << std::endl;
	std::cout << "line vertex shader: " << lineShader->vertexShader << std::endl;
	std::cout << "line fragment shader: " << lineShader->fragmentShader << std::endl;

	std::cout << "number of materials: " << manager->getNumberOfAssets<Material>() << " number of shaders: " << manager->getNumberOfAssets<Shader>() << std::endl; 
}

void LogicSystem::update()
{
	//std::cout << "LogicSystem update called" << std::endl;

	//std::cout << "line material guid: " << lineMaterial->assetId.toString() << "shader id: " << lineMaterial->shaderId.toString() << std::endl;

	if(Input::getKeyDown(KeyCode::I)){
		Entity* entity = manager->instantiate();
		if(entity != NULL){
			//std::cout << "AAAA" << std::endl;
			//std::cout << "Creating new entity with id: " << entity->entityId.toString() << " total entities now: " << manager->getNumberOfEntities() << " total number of line renderers: " << manager->getNumberOfComponents<LineRenderer>() << std::endl;

			Transform* transform = entity->addComponent<Transform>();
			LineRenderer* line = entity->addComponent<LineRenderer>();
			line->materialId = lineMaterial->assetId;
			std::cout << "Creating new entity with id: " << entity->entityId.toString() << " total entities now: " << manager->getNumberOfEntities() << " creating line renderer: " << line->componentId.toString() << " total number of line renderers: " << manager->getNumberOfComponents<LineRenderer>() << " start: " << line->start.x << " " << line->start.y << " " << line->start.z << " end: " << line->end.x << " " << line->end.y << " " << line->end.z << std::endl;
		}
	}
}