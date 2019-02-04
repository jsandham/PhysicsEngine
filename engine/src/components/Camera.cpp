#include "../../include/components/Camera.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Camera::Camera()
{
	position = glm::vec3(0.0f, 2.0f, 0.0f);
	front = glm::vec3(0.0f, 0.0f, -1.0f);
	up = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::normalize(glm::cross(front, up));

	worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

	lastPosX = 0;
	lastPosY = 0;

	currentPosX = 0;
	currentPosY = 0;

	enabled = true;
	priority = 10;

	x = 0;
	y = 0;
	width = 1000;
	height = 1000;

	projection = glm::perspective(45.0f, 1.0f * 1000 / 1000, 0.1f, 100.0f);
	view = glm::lookAt(position, position + front, up);

	frustum.setPerspective(45.0f, 1.0f * 1000 / 1000, 0.1f, 100.0f);

	frustum.setCamera(position, front, up, right);

	backgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
}

Camera::Camera(unsigned char* data)
{
	
}

Camera::~Camera()
{

}

void* Camera::operator new(size_t size)
{
	return getAllocator<Camera>().allocate();
}

void Camera::operator delete(void*)
{

}

void Camera::load(CameraData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	position = data.position;
	backgroundColor = data.backgroundColor;
}

glm::vec3& Camera::getPosition() 
{
	return position;
}

glm::vec3& Camera::getFront() 
{
	return front;
}

glm::vec3& Camera::getUp() 
{
	return up;
}


glm::vec3& Camera::getRight() 
{
	return right;
}

glm::vec3& Camera::getWorldUp()
{
	return worldUp;
}


glm::mat4& Camera::getViewMatrix()
{
	view = glm::lookAt(position, position + front, up);
	return view;
}


glm::mat4& Camera::getProjMatrix()
{
	return projection;
}

glm::vec4& Camera::getBackgroundColor()
{
	return backgroundColor;
}


void Camera::setPosition(glm::vec3& position)
{
	this->position = position;
}


void Camera::setFront(glm::vec3& front)
{
	this->front = front;
}


void Camera::setUp(glm::vec3& up)
{
	this->up = up;
}

void Camera::setRight(glm::vec3& right)
{
	this->right = right;
}

void Camera::setProjMatrix(glm::mat4& projection)
{
	this->projection = projection;
}

void Camera::setBackgroundColor(glm::vec4& backgroundColor)
{
	this->backgroundColor = backgroundColor;
}


int Camera::checkPointInFrustum(glm::vec3 point)
{
	return frustum.checkPoint(point);
}


int Camera::checkSphereInFrustum(glm::vec3 centre, float radius)
{
	return frustum.checkSphere(centre, radius);
}


int Camera::checkAABBInFrustum(glm::vec3 min, glm::vec3 max)
{
	return frustum.checkAABB(min, max);
}