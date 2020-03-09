#include "../../include/components/Camera.h"

using namespace PhysicsEngine;

float Plane::distance(glm::vec3 point) const
{
	float d = -glm::dot(n,x0);

	return (glm::dot(n,point) + d) / sqrt(glm::dot(n,n));
}

float Viewport::getAspectRatio() const
{
	return (float)height / (float)width;
}

int Frustum::checkPoint(glm::vec3 point) const
{
	// loop over all 6 planes
	for(int i = 0; i < 6; i++){
		if(planes[i].distance(point) < 0){
			return -1; // outside
		}
	}

	return 1; // inside
}

int Frustum::checkSphere(glm::vec3 centre, float radius) const
{
	bool answer = 1;

	// loop over all 6 planes
	for(int i = 0; i < 6; i++){
		float distance = planes[i].distance(centre);
		if(distance < -radius){
			return -1; // outside
		}
		else if(distance < radius){
			answer = 0; // intersect
		}
	}

	return answer; // inside
}

int Frustum::checkAABB(glm::vec3 min, glm::vec3 max) const
{
	return 1;
}

Camera::Camera()
{
	componentId = Guid::INVALID;
	entityId = Guid::INVALID;
	targetTextureId = Guid::INVALID;

	mainFBO = 0;
	colorTex = 0;
	depthTex = 0;

	geometryFBO = 0;
	positionTex = 0;
	normalTex = 0;

	mode = CameraMode::Main;

	viewport.x = 0;
	viewport.y = 0;
	viewport.width = 1024;
	viewport.height = 1024;

	frustum.fov = 45.0f;
	frustum.nearPlane = 0.1f;
	frustum.farPlane = 250.0f;

	position = glm::vec3(0.0f, 2.0f, 0.0f);
	front = glm::vec3(0.0f, 0.0f, -1.0f);
	up = glm::vec3(0.0f, 1.0f, 0.0f);
	backgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

	isCreated = false;
	useSSAO = false;

	updateInternalCameraState();
}

Camera::Camera(std::vector<char> data)
{
	deserialize(data);

	updateInternalCameraState();
}

Camera::~Camera()
{

}

std::vector<char> Camera::serialize() const
{
	return serialize(componentId, entityId);
}

std::vector<char> Camera::serialize(Guid componentId, Guid entityId) const
{
	CameraHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.targetTextureId = targetTextureId;
	header.mode = mode;
	header.position = position;
	header.front = front;
	header.up = up;
	header.backgroundColor = backgroundColor;
	header.x = viewport.x;
	header.y = viewport.y;
	header.width = viewport.width;
	header.height = viewport.height;
	header.fov = frustum.fov;
	header.nearPlane = frustum.nearPlane;
	header.farPlane = frustum.farPlane;

	std::vector<char> data(sizeof(CameraHeader));

	memcpy(&data[0], &header, sizeof(CameraHeader));

	return data;
}

void Camera::deserialize(std::vector<char> data)
{
	CameraHeader* header = reinterpret_cast<CameraHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	targetTextureId = header->targetTextureId;

	mode = header->mode;

	viewport.x = header->x;
	viewport.y = header->y;
	viewport.width = header->width;
	viewport.height = header->height;

	frustum.fov = header->fov;
	frustum.nearPlane = header->nearPlane;
	frustum.farPlane = header->farPlane;

	position = header->position;
	front = header->front;
	up = header->up;
	backgroundColor = header->backgroundColor;
}

void Camera::updateInternalCameraState()
{
	this->front = glm::normalize(front);
	this->up = glm::normalize(up);
	this->right = glm::normalize(glm::cross(front, up));

	float tan = (float)glm::tan(glm::radians(0.5f * frustum.fov));
	float nearPlaneHeight = frustum.nearPlane * tan;
	float nearPlaneWidth = viewport.getAspectRatio() * nearPlaneHeight;
	float farPlaneHeight = frustum.farPlane * tan;
	float farPlaneWidth = viewport.getAspectRatio() * farPlaneHeight;

	// far and near plane intersection along front line
	glm::vec3 fc = position + frustum.farPlane * front;
	glm::vec3 nc = position + frustum.nearPlane * front;

	frustum.planes[NEAR].n = front;
	frustum.planes[NEAR].x0 = nc;

	frustum.planes[FAR].n = -front;
	frustum.planes[FAR].x0 = fc;

	glm::vec3 temp;

	temp = (nc + nearPlaneHeight*up) - position;
	temp = glm::normalize(temp);
	frustum.planes[TOP].n = glm::cross(temp, right);
	frustum.planes[TOP].x0 = nc + nearPlaneHeight*up;

	temp = (nc - nearPlaneHeight*up) - position;
	temp = glm::normalize(temp);
	frustum.planes[BOTTOM].n = -glm::cross(temp, right);
	frustum.planes[BOTTOM].x0 = nc - nearPlaneHeight*up;

	temp = (nc - nearPlaneWidth*right) - position;
	temp = glm::normalize(temp);
	frustum.planes[LEFT].n = glm::cross(temp, up);
	frustum.planes[LEFT].x0 = nc - nearPlaneWidth*right;

	temp = (nc + nearPlaneWidth*right) - position;
	temp = glm::normalize(temp);
	frustum.planes[RIGHT].n = -glm::cross(temp, up);
	frustum.planes[RIGHT].x0 = nc + nearPlaneWidth*right;
}

glm::mat4 Camera::getViewMatrix() const
{
	return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjMatrix() const
{
	return glm::perspective(glm::radians(frustum.fov), viewport.getAspectRatio(), frustum.nearPlane, frustum.farPlane);
}

int Camera::checkPointInFrustum(glm::vec3 point) const
{
	return frustum.checkPoint(point);
}

int Camera::checkSphereInFrustum(glm::vec3 centre, float radius) const
{
	return frustum.checkSphere(centre, radius);
}

int Camera::checkAABBInFrustum(glm::vec3 min, glm::vec3 max) const
{
	return frustum.checkAABB(min, max);
}