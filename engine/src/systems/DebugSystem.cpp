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

				glm::vec4 rayClip = glm::vec4(screen_x, screen_y, -1.0f, 1.0f);
				glm::vec4 rayEye = glm::inverse(projection) * rayClip;
				rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
				glm::vec3 rayWorld = glm::vec3((glm::inverse(view) * rayEye));
				rayWorld = glm::normalize(rayWorld);

				std::cout << "ray world x: " << rayWorld.x << " " << rayWorld.y << " " << rayWorld.z << " camera position: " << camera->position.x << " " << camera->position.y << " " << camera->position.z << std::endl;

				//glm::vec4 nearPoint = projViewInv * glm::vec4(screen_x, screen_y, 0, 1);
				//glm::vec4 farPoint = projViewInv * glm::vec4(screen_x, screen_y, 1, 1);

				lineRenderer->start = camera->position;//glm::vec3(nearPoint.x / nearPoint.w, nearPoint.y / nearPoint.w, nearPoint.z / nearPoint.w);
				lineRenderer->end = camera->position + 10.0f * rayWorld;//glm::vec3(farPoint.x / farPoint.w, farPoint.y / farPoint.w, farPoint.z / farPoint.w);

				std::cout << "start: " << lineRenderer->start.x << " " << lineRenderer->start.y << " " << lineRenderer->start.z << "  end: " << lineRenderer->end.x << " " << lineRenderer->end.y << " " << lineRenderer->end.z << std::endl;

				Collider* hitCollider = NULL;
				if(manager->raycast(lineRenderer->start, rayWorld, 100.0f, &hitCollider))
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