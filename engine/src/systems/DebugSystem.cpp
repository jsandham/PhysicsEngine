#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/World.h"

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

DebugSystem::DebugSystem(std::vector<char> data)
{
	type = 3;
}

DebugSystem::~DebugSystem()
{
	
}

void DebugSystem::init(World* world)
{
	this->world = world;

	lineMaterial = world->createAsset<Material>();
	lineShader = world->createAsset<Shader>();

	lineShader->vertexShader = Shader::lineVertexShader;
	lineShader->fragmentShader = Shader::lineFragmentShader;
	
	lineShader->compile();
	
	lineMaterial->shaderId = lineShader->assetId;
}

void DebugSystem::update(Input input)
{
	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	if(getKeyDown(input, KeyCode::D)){
		world->debug = !world->debug;
	}

	if(world->debug){
		if (getKeyDown(input, KeyCode::I)){
			Entity* entity = world->instantiate();
			if(entity != NULL){
				Transform* transform = entity->addComponent<Transform>(world);
				LineRenderer* lineRenderer = entity->addComponent<LineRenderer>(world);

				lineRenderer->materialId = lineMaterial->assetId;

				glm::mat4 view = camera->getViewMatrix();
				glm::mat4 projection = camera->getProjMatrix();
				glm::mat4 projViewInv = glm::inverse(projection * view);
		
				int x = input.mousePosX;
				int y = input.mousePosY;
				int width = camera->width;
				int height = camera->height - 40;

				float screen_x = (x - 0.5f * width) / (0.5f * width);
				float screen_y = (0.5f * height - y) / (0.5f * height);

				std::cout << "x: " << x << " y: " << y << " width: " << width << " height: " << height << " screen_x: " << screen_x << " screen_y: " << screen_y << std::endl;

				glm::vec4 rayClip = glm::vec4(screen_x, screen_y, -1.0f, 1.0f);
				glm::vec4 rayEye = glm::inverse(projection) * rayClip;
				rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
				glm::vec3 rayWorld = glm::vec3((glm::inverse(view) * rayEye));
				rayWorld = glm::normalize(rayWorld);

				lineRenderer->start = camera->position;
				lineRenderer->end = camera->position + 10.0f * rayWorld;

				Collider* hitCollider = NULL;
				if(world->raycast(lineRenderer->start, rayWorld, 100.0f, &hitCollider))
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