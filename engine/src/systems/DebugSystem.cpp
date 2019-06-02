#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"

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
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if(type != 3){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
}

DebugSystem::~DebugSystem()
{
	
}

void DebugSystem::init(World* world)
{
	this->world = world;

	std::vector<float> lines;
	lines.resize(6, 0.0f);

	buffer.init(lines);

	colorShader = world->createAsset<Shader>();
	colorShader->vertexShader = Shader::colorVertexShader;
	colorShader->fragmentShader = Shader::colorFragmentShader;
	colorShader->compile();

	colorMat = world->createAsset<Material>();
	colorMat->shaderId = colorShader->assetId;
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

	if(world->debug){
		if (getKeyDown(input, KeyCode::I)){

			glm::mat4 view = camera->getViewMatrix();
			glm::mat4 projection = camera->getProjMatrix();
			glm::mat4 projViewInv = glm::inverse(projection * view);
	
			int x = input.mousePosX;
			int y = input.mousePosY;
			int width = camera->viewport.width;
			int height = camera->viewport.height - 40;

			float screen_x = (x - 0.5f * width) / (0.5f * width);
			float screen_y = (0.5f * height - y) / (0.5f * height);

			std::cout << "x: " << x << " y: " << y << " width: " << width << " height: " << height << " screen_x: " << screen_x << " screen_y: " << screen_y << std::endl;

			glm::vec4 rayClip = glm::vec4(screen_x, screen_y, -1.0f, 1.0f);
			glm::vec4 rayEye = glm::inverse(projection) * rayClip;
			rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
			glm::vec3 rayWorld = glm::vec3((glm::inverse(view) * rayEye));
			rayWorld = glm::normalize(rayWorld);

			glm::vec3 start = camera->position;
			glm::vec3 end = camera->position + 10.0f * rayWorld;

			std::vector<float> temp;
			temp.resize(6);
			temp[0] = start.x;
			temp[1] = start.y;
			temp[2] = start.z;
			temp[3] = end.x;
			temp[4] = end.y;
			temp[5] = end.z;
			buffer.update(temp);

			Collider* hitCollider = NULL;
			if(world->raycast(start, rayWorld, 100.0f, &hitCollider))
			{
				if(hitCollider == NULL){
					std::cout << "Raycast hit sphere collider but reported hit collider as NULL???" << std::endl;
				}
				else
				{
					std::cout << "Raycast hit sphere collider: " << hitCollider->componentId.toString() << " entity id: " << hitCollider->entityId.toString() << std::endl;

					MeshRenderer* meshRenderer = world->getComponent<MeshRenderer>(hitCollider->entityId);
					if(meshRenderer != NULL){
						std::cout << "Setting material to " << colorMat->assetId.toString() << " mesh renderer is static? " << meshRenderer->isStatic << std::endl;
						meshRenderer->materialId = colorMat->assetId;
						meshRenderer->isStatic = false;
					}
				}
			}
			else
			{
				std::cout << "Raycast missed!!!" << std::endl;
			}
		}
	}

	Graphics::render(world, &buffer.shader, ShaderVariant::None, glm::mat4(1.0f), buffer.VAO, GL_LINES, (GLsizei)buffer.size / 3, NULL);
}