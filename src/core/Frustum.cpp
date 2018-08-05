#include <iostream>
#include "Frustum.h"

using namespace PhysicsEngine;

#define RADIAN 3.14159265358979323846f / 180.0f


float Plane::distance(glm::vec3 point)
{
	float d = -glm::dot(n,x0);

	return (glm::dot(n,point) + d) / sqrt(glm::dot(n,n));
}

Frustum::Frustum()
{
}

Frustum::~Frustum()
{

}

void Frustum::setPerspective(float angle, float ratio, float near, float far)
{
	this->angle = angle;
	this->ratio = ratio;
	this->near = near;
	this->far = far;

	float tan = (float)glm::tan(0.5f*RADIAN*angle);
	nearPlaneHeight = near * tan;
	nearPlaneWidth = ratio * nearPlaneHeight;
	farPlaneHeight = far * tan;
	farPlaneWidth = farPlaneHeight * ratio;
}

void Frustum::setCamera(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right)
{
	this->position = position;
	this->front = glm::normalize(front);
	this->up = glm::normalize(up);
	this->right = glm::normalize(right);

	// far and near plane intersection along front line
	glm::vec3 fc = position + far * front;
	glm::vec3 nc = position + near * front;

	planes[NEAR].n = front;
	planes[NEAR].x0 = nc;

	planes[FAR].n = -front;
	planes[FAR].x0 = fc;

	glm::vec3 temp;

	temp = (nc + nearPlaneHeight*up) - position;
	temp = glm::normalize(temp);
	planes[TOP].n = glm::cross(temp, right);
	planes[TOP].x0 = nc + nearPlaneHeight*up;

	temp = (nc - nearPlaneHeight*up) - position;
	temp = glm::normalize(temp);
	planes[BOTTOM].n = -glm::cross(temp, right);
	planes[BOTTOM].x0 = nc - nearPlaneHeight*up;

	temp = (nc - nearPlaneWidth*right) - position;
	temp = glm::normalize(temp);
	planes[LEFT].n = glm::cross(temp, up);
	planes[LEFT].x0 = nc - nearPlaneWidth*right;

	temp = (nc + nearPlaneWidth*right) - position;
	temp = glm::normalize(temp);
	planes[RIGHT].n = -glm::cross(temp, up);
	planes[RIGHT].x0 = nc + nearPlaneWidth*right;


	// std::cout << "near normal: " << planes[NEAR].n.x << " " << planes[NEAR].n.y << " " << planes[NEAR].n.z << std::endl;
	// std::cout << "far normal: " << planes[FAR].n.x << " " << planes[FAR].n.y << " " << planes[FAR].n.z << std::endl;
	// std::cout << "top normal: " << planes[TOP].n.x << " " << planes[TOP].n.y << " " << planes[TOP].n.z << std::endl;
	// std::cout << "bottom normal: " << planes[BOTTOM].n.x << " " << planes[BOTTOM].n.y << " " << planes[BOTTOM].n.z << std::endl;
	// std::cout << "left normal: " << planes[LEFT].n.x << " " << planes[LEFT].n.y << " " << planes[LEFT].n.z << std::endl;
	// std::cout << "right normal: " << planes[RIGHT].n.x << " " << planes[RIGHT].n.y << " " << planes[RIGHT].n.z << std::endl;
}

void Frustum::setCameraSlow(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right)
{
	setCamera(position, front, up, right);

	// far and near plane intersection along front line
	glm::vec3 fc = position + far * front;
	glm::vec3 nc = position + near * front;

	// vertices for near plane
	glm::vec3 ntl = nc + 0.5f*nearPlaneHeight*up - 0.5f*nearPlaneWidth*right;
	glm::vec3 ntr = nc + 0.5f*nearPlaneHeight*up + 0.5f*nearPlaneWidth*right;
	glm::vec3 nbl = nc - 0.5f*nearPlaneHeight*up - 0.5f*nearPlaneWidth*right;
	glm::vec3 nbr = nc - 0.5f*nearPlaneHeight*up + 0.5f*nearPlaneWidth*right;

	// vertices for far plane
	glm::vec3 ftl = fc + 0.5f*farPlaneHeight*up - 0.5f*farPlaneWidth*right;
	glm::vec3 ftr = fc + 0.5f*farPlaneHeight*up + 0.5f*farPlaneWidth*right;
	glm::vec3 fbl = fc - 0.5f*farPlaneHeight*up - 0.5f*farPlaneWidth*right;
	glm::vec3 fbr = fc - 0.5f*farPlaneHeight*up + 0.5f*farPlaneWidth*right;

	// top plane triangles
	frustumVertices[0] = ftl;
	frustumVertices[1] = ftr;
	frustumVertices[2] = ntl;
	frustumVertices[3] = ftr;
	frustumVertices[4] = ntr;
	frustumVertices[5] = ntl;

	// bottom plane triangles
	frustumVertices[6] = fbl;
	frustumVertices[7] = fbr;
	frustumVertices[8] = nbl;
	frustumVertices[9] = fbr;
	frustumVertices[10] = nbr;
	frustumVertices[11] = nbl;

	// left plane triangles
	frustumVertices[12] = ftl;
	frustumVertices[13] = nbl;
	frustumVertices[14] = fbl;
	frustumVertices[15] = ftl;
	frustumVertices[16] = ntl;
	frustumVertices[17] = nbl;

	// right plane triangles
	frustumVertices[18] = ftr;
	frustumVertices[19] = nbr;
	frustumVertices[20] = fbr;
	frustumVertices[21] = ftr;
	frustumVertices[22] = ntr;
	frustumVertices[23] = nbr;

	// near plane triangles
	frustumVertices[24] = ntl;
	frustumVertices[25] = ntr;
	frustumVertices[26] = nbr;
	frustumVertices[27] = ntl;
	frustumVertices[28] = nbr;
	frustumVertices[29] = nbl;

	// far plane triangles
	frustumVertices[30] = ftl;
	frustumVertices[31] = ftr;
	frustumVertices[32] = fbr;
	frustumVertices[33] = ftl;
	frustumVertices[34] = fbr;
	frustumVertices[35] = fbl;

	for(int i = 0; i < 6; i++){
		glm::vec3 n = planes[i].n;
		for(int j = 0; j < 6; j++){
			frustumNormals[6*i + j] = n;
		}
	}
}

int Frustum::checkPoint(glm::vec3 point)
{
	// loop over all 6 planes
	for(int i = 0; i < 6; i++){
		if(planes[i].distance(point) < 0){
			return -1; // outside
		}
	}

	return 1; // inside
}

int Frustum::checkSphere(glm::vec3 centre, float radius)
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

int Frustum::checkAABB(glm::vec3 min, glm::vec3 max)
{
	return 1;
}