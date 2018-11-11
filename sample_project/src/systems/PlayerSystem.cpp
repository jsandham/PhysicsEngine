#include "../../include/systems/PlayerSystem.h"

#include <components/Camera.h>

#include <core/Input.h>
#include <core/Time.h>
#include <core/Log.h>
#include <core/Manager.h>

using namespace PhysicsEngine;

const float PlayerSystem::PAN_SENSITIVITY = 0.01f;
const float PlayerSystem::SCROLL_SENSITIVITY = 0.001f;
const float PlayerSystem::TRANSLATE_SENSITIVITY = 0.05f;

PlayerSystem::PlayerSystem()
{
	type = 11;
}

PlayerSystem::PlayerSystem(unsigned char* data)
{
	type = 11;
}

PlayerSystem::~PlayerSystem()
{

}

void PlayerSystem::init()
{
	
}

void PlayerSystem::update()
{
	Camera* camera = manager->getComponentByIndex<Camera>(0);

	glm::vec3 position = camera->getPosition();
	glm::vec3 front = camera->getFront();
	glm::vec3 up = camera->getUp();
	glm::vec3 right = camera->getRight();

	int lastPosX = camera->lastPosX;
	int lastPosY = camera->lastPosY;

	int currentPosX = camera->currentPosX;
	int currentPosY = camera->currentPosY;

	if (Input::getKey(KeyCode::Up)){
		position += up * PlayerSystem::TRANSLATE_SENSITIVITY;
	}
	if (Input::getKey(KeyCode::Down)){
		position -= up * PlayerSystem::TRANSLATE_SENSITIVITY;
	}
	if (Input::getKey(KeyCode::Left)){
		position -= right * PlayerSystem::TRANSLATE_SENSITIVITY;
	}
	if (Input::getKey(KeyCode::Right)){
		position += right * PlayerSystem::TRANSLATE_SENSITIVITY;
	}

	glm::vec2 mouseDelta = glm::vec2(0.0f, 0.0f);

	position += PlayerSystem::SCROLL_SENSITIVITY * Input::getMouseDelta() * front;

	if (Input::getMouseButtonDown(LButton)){
		currentPosX = Input::getMousePosX();
		currentPosY = Input::getMousePosY();
	}
	else if (Input::getMouseButton(LButton)){
		currentPosX = Input::getMousePosX();
		currentPosY = Input::getMousePosY();
		mouseDelta.x = PlayerSystem::PAN_SENSITIVITY * (currentPosX - lastPosX);
		mouseDelta.y = PlayerSystem::PAN_SENSITIVITY * (currentPosY - lastPosY);
	}
	else if (Input::getMouseButtonUp(LButton)){
		mouseDelta = glm::vec2(0.0f, 0.0f);
	}

	lastPosX = currentPosX;
	lastPosY = currentPosY;

	// rotation around the camera up vector
	front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.x, up)) * front;
	front = glm::normalize(front);

	// rotation around the camera right vector
	front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.y, right)) * front;
	front = glm::normalize(front);

	right = glm::normalize(glm::cross(front, up));
	//right = glm::normalize(glm::cross(front, worldUp));

	up = glm::normalize(glm::cross(right, front));

	camera->setPosition(position);
	camera->setFront(front);
	camera->setUp(up);
	camera->setRight(right);

	camera->lastPosX = lastPosX;
	camera->lastPosY = lastPosY;
	camera->currentPosX = currentPosX;
	camera->currentPosY = currentPosY;














	/*std::vector<FPSCamera*> cameras = manager->getFPSCameras();

	for (unsigned int i = 0; i < cameras.size(); i++){

		FPSCamera *camera = cameras[i];

		glm::vec3 position = camera->getPosition();
		glm::vec3 front = camera->getFront();
		glm::vec3 up = camera->getUp();
		glm::vec3 right = camera->getRight();
		glm::vec3 worldUp = camera->getWorldUp();

		int2 lastPos = camera->getLastPosition();
		int2 currentPos = camera->getCurrentPosition();

		float yaw = camera->getYaw();
		float pitch = camera->getPitch();

		float translateSpeed = 0.03f;
		if (Input::getKey(sf::Keyboard::LShift)){
			translateSpeed = 0.06f;
		}

		if (Input::getKey(sf::Keyboard::Up) || Input::getKey(sf::Keyboard::W)){
			position += translateSpeed * glm::normalize(glm::vec3(front.x, 0.0f, front.z));
		}
		if (Input::getKey(sf::Keyboard::Down) || Input::getKey(sf::Keyboard::S)){
			position -= translateSpeed * glm::normalize(glm::vec3(front.x, 0.0f, front.z));
		}
		if (Input::getKey(sf::Keyboard::Left) || Input::getKey(sf::Keyboard::A)){
			position -= right * translateSpeed;
		}
		if (Input::getKey(sf::Keyboard::Right) || Input::getKey(sf::Keyboard::D)){
			position += right * translateSpeed;
		}
	
		currentPos = Input::getMousePosition();

		float offsetx = (float)(currentPos.x - lastPos.x);
		float offsety = (float)(lastPos.y - currentPos.y);

		lastPos.x = currentPos.x;
		lastPos.y = currentPos.y;

		yaw += 0.4f * offsetx;
		pitch += 0.4f * offsety;

		yaw = glm::mod(yaw, 360.0f);

		if (pitch > 89.0f){
			pitch = 89.0f;
		}
		if (pitch < -89.0f){
			pitch = -89.0f;
		}

		glm::vec3 newFront;
		newFront.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
		newFront.y = sin(glm::radians(pitch));
		newFront.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
		front = glm::normalize(newFront);

		right = glm::normalize(glm::cross(front, worldUp));

		up = glm::normalize(glm::cross(right, front));

		camera->setPosition(position);
		camera->setFront(front);
		camera->setUp(up);
		camera->setRight(right);

		camera->setLastPosition(lastPos);
		camera->setCurrentPosition(currentPos);

		camera->setYaw(yaw);
		camera->setPitch(pitch);
	}*/
}
