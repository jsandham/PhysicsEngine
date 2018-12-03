#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/Input.h"
#include "../../include/core/Manager.h"

#include "../../include/components/Transform.h"
#include "../../include/components/LineRenderer.h"

#include "../../include/graphics/Graphics.h"

#include "../../include/glm/glm.hpp"
#include "../../include/glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

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
	lineShader = manager->create<Shader>();

	lineShader->vertexShader = Shader::lineVertexShader;
	lineShader->fragmentShader = Shader::lineFragmentShader;
	
	lineShader->compile();
	
	lineMaterial->shaderId = lineShader->assetId;
}

void DebugSystem::update()
{
	Camera* camera;
	if(manager->getNumberOfComponents<Camera>() > 0){
		camera = manager->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	if(Input::getKeyDown(KeyCode::D)){
		manager->debug = !manager->debug;
	}

	if(manager->debug){
		if (Input::getKeyDown(KeyCode::I)){
			Entity* entity = manager->instantiate();
			if(entity != NULL){
				Transform* transform = entity->addComponent<Transform>();
				LineRenderer* lineRenderer = entity->addComponent<LineRenderer>();

				lineRenderer->materialId = lineMaterial->assetId;

				glm::mat4 view = camera->getViewMatrix();
				glm::mat4 projection = camera->getProjMatrix();
				glm::mat4 projViewInv = glm::inverse(projection * view);
		
				int x = Input::getMousePosX();
				int y = Input::getMousePosY();
				int width = camera->width;
				int height = camera->height;

				float screen_x = (x - 0.5f * width) / (0.5f * width);
				float screen_y = (0.5f * height - y) / (0.5f * height);

				glm::vec4 nearPoint = projViewInv * glm::vec4(screen_x, screen_y, 0, 1);
				glm::vec4 farPoint = projViewInv * glm::vec4(screen_x, screen_y, 1, 1);

				lineRenderer->start = glm::vec3(nearPoint.x / nearPoint.w, nearPoint.y / nearPoint.w, nearPoint.z / nearPoint.w);
				lineRenderer->end = glm::vec3(farPoint.x / farPoint.w, farPoint.y / farPoint.w, farPoint.z / farPoint.w);

				Collider* hitCollider = NULL;
				if(manager->raycast(lineRenderer->start, lineRenderer->end - lineRenderer->start, 100.0f, &hitCollider))
				{
					if(hitCollider == NULL){
						std::cout << "Raycast hit sphere collider but reported hit collider as NULL???" << std::endl;
					}
					else
					{
						std::cout << "Raycast hit sphere collider: " << hitCollider->componentId.toString() << std::endl;
					}
				}
				else
				{
					std::cout << "Raycast missed!!!" << std::endl;
				}
			}
		}
	}
}