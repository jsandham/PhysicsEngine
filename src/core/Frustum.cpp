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

	frustumVertices.clear();
	frustumNormals.clear();

	// top plane triangles
	frustumVertices.push_back(ftl.x);
	frustumVertices.push_back(ftl.y);
	frustumVertices.push_back(ftl.z);
	frustumVertices.push_back(ftr.x);
	frustumVertices.push_back(ftr.y);
	frustumVertices.push_back(ftr.z);
	frustumVertices.push_back(ntl.x);
	frustumVertices.push_back(ntl.y);
	frustumVertices.push_back(ntl.z);
	frustumVertices.push_back(ftr.x);
	frustumVertices.push_back(ftr.y);
	frustumVertices.push_back(ftr.z);
	frustumVertices.push_back(ntr.x);
	frustumVertices.push_back(ntr.y);
	frustumVertices.push_back(ntr.z);
	frustumVertices.push_back(ntl.x);
	frustumVertices.push_back(ntl.y);
	frustumVertices.push_back(ntl.z);

	// bottom plane triangles
	frustumVertices.push_back(fbl.x);
	frustumVertices.push_back(fbl.y);
	frustumVertices.push_back(fbl.z);
	frustumVertices.push_back(fbr.x);
	frustumVertices.push_back(fbr.y);
	frustumVertices.push_back(fbr.z);
	frustumVertices.push_back(nbl.x);
	frustumVertices.push_back(nbl.y);
	frustumVertices.push_back(nbl.z);
	frustumVertices.push_back(fbr.x);
	frustumVertices.push_back(fbr.y);
	frustumVertices.push_back(fbr.z);
	frustumVertices.push_back(nbr.x);
	frustumVertices.push_back(nbr.y);
	frustumVertices.push_back(nbr.z);
	frustumVertices.push_back(nbl.x);
	frustumVertices.push_back(nbl.y);
	frustumVertices.push_back(nbl.z);

	// left plane triangles
	frustumVertices.push_back(ftl.x);
	frustumVertices.push_back(ftl.y);
	frustumVertices.push_back(ftl.z);
	frustumVertices.push_back(nbl.x);
	frustumVertices.push_back(nbl.y);
	frustumVertices.push_back(nbl.z);
	frustumVertices.push_back(fbl.x);
	frustumVertices.push_back(fbl.y);
	frustumVertices.push_back(fbl.z);
	frustumVertices.push_back(ftl.x);
	frustumVertices.push_back(ftl.y);
	frustumVertices.push_back(ftl.z);
	frustumVertices.push_back(ntl.x);
	frustumVertices.push_back(ntl.y);
	frustumVertices.push_back(ntl.z);
	frustumVertices.push_back(nbl.x);
	frustumVertices.push_back(nbl.y);
	frustumVertices.push_back(nbl.z);

	// right plane triangles
	frustumVertices.push_back(ftr.x);
	frustumVertices.push_back(ftr.y);
	frustumVertices.push_back(ftr.z);
	frustumVertices.push_back(nbr.x);
	frustumVertices.push_back(nbr.y);
	frustumVertices.push_back(nbr.z);
	frustumVertices.push_back(fbr.x);
	frustumVertices.push_back(fbr.y);
	frustumVertices.push_back(fbr.z);
	frustumVertices.push_back(ftr.x);
	frustumVertices.push_back(ftr.y);
	frustumVertices.push_back(ftr.z);
	frustumVertices.push_back(ntr.x);
	frustumVertices.push_back(ntr.y);
	frustumVertices.push_back(ntr.z);
	frustumVertices.push_back(nbr.x);
	frustumVertices.push_back(nbr.y);
	frustumVertices.push_back(nbr.z);

	// near plane triangles
	frustumVertices.push_back(ntl.x);
	frustumVertices.push_back(ntl.y);
	frustumVertices.push_back(ntl.z);
	frustumVertices.push_back(ntr.x);
	frustumVertices.push_back(ntr.y);
	frustumVertices.push_back(ntr.z);
	frustumVertices.push_back(nbr.x);
	frustumVertices.push_back(nbr.y);
	frustumVertices.push_back(nbr.z);
	frustumVertices.push_back(ntl.x);
	frustumVertices.push_back(ntl.y);
	frustumVertices.push_back(ntl.z);
	frustumVertices.push_back(nbr.x);
	frustumVertices.push_back(nbr.y);
	frustumVertices.push_back(nbr.z);
	frustumVertices.push_back(nbl.x);
	frustumVertices.push_back(nbl.y);
	frustumVertices.push_back(nbl.z);

	// far plane triangles
	frustumVertices.push_back(ftl.x);
	frustumVertices.push_back(ftl.y);
	frustumVertices.push_back(ftl.z);
	frustumVertices.push_back(ftr.x);
	frustumVertices.push_back(ftr.y);
	frustumVertices.push_back(ftr.z);
	frustumVertices.push_back(fbr.x);
	frustumVertices.push_back(fbr.y);
	frustumVertices.push_back(fbr.z);
	frustumVertices.push_back(ftl.x);
	frustumVertices.push_back(ftl.y);
	frustumVertices.push_back(ftl.z);
	frustumVertices.push_back(fbr.x);
	frustumVertices.push_back(fbr.y);
	frustumVertices.push_back(fbr.z);
	frustumVertices.push_back(fbl.x);
	frustumVertices.push_back(fbl.y);
	frustumVertices.push_back(fbl.z);

	for(int i = 0; i < 6; i++){
		glm::vec3 n = planes[i].n;
		for(int j = 0;j < 6; j++){
			frustumNormals.push_back(n.x);
			frustumNormals.push_back(n.y);
			frustumNormals.push_back(n.z);
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

std::vector<float> Frustum::getTriVertices()
{
	return frustumVertices;
}


std::vector<float> Frustum::getTriNormals()
{
	return frustumNormals;
}

