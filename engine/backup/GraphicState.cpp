// #include <iostream>
// #include "../../include/graphics/GraphicState.h"

// using namespace PhysicsEngine;

// GraphicState::GraphicState()
// {

// }

// GraphicState::~GraphicState()
// {

// }

// void GraphicState::init()
// {
// 	buffers[(int)CameraBuffer].generate(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
// 	buffers[(int)CameraBuffer].bind();
// 	buffers[(int)CameraBuffer].setData(NULL, 144);
// 	buffers[(int)CameraBuffer].setRange((int)CameraBuffer, 0, 144);
// 	buffers[(int)CameraBuffer].unbind();

// 	buffers[(int)ShadowBuffer].generate(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
// 	buffers[(int)ShadowBuffer].bind();
// 	buffers[(int)ShadowBuffer].setData(NULL, 736);
// 	buffers[(int)ShadowBuffer].setRange((int)ShadowBuffer, 0, 736);
// 	buffers[(int)ShadowBuffer].unbind();

// 	buffers[(int)DirectionalLightBuffer].generate(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
// 	buffers[(int)DirectionalLightBuffer].bind();
// 	buffers[(int)DirectionalLightBuffer].setData(NULL, 64);
// 	buffers[(int)DirectionalLightBuffer].setRange((int)DirectionalLightBuffer, 0, 64);
// 	buffers[(int)DirectionalLightBuffer].unbind();

// 	buffers[(int)SpotLightBuffer].generate(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
// 	buffers[(int)SpotLightBuffer].bind();
// 	buffers[(int)SpotLightBuffer].setData(NULL, 100);
// 	buffers[(int)SpotLightBuffer].setRange((int)SpotLightBuffer, 0, 100);
// 	buffers[(int)SpotLightBuffer].unbind();

// 	buffers[(int)PointLightBuffer].generate(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
// 	buffers[(int)PointLightBuffer].bind();
// 	buffers[(int)PointLightBuffer].setData(NULL, 76);
// 	buffers[(int)PointLightBuffer].setRange((int)PointLightBuffer, 0, 76);
// 	buffers[(int)PointLightBuffer].unbind();

// 	shadowTexture2D = NULL;
// 	shadowCubemap = NULL;
// }

// void GraphicState::bind(UniformBuffer buffer)
// {
// 	buffers[(int)buffer].bind();
// }

// void GraphicState::unbind(UniformBuffer buffer)
// {
// 	buffers[(int)buffer].unbind();
// }


// // setters

// void GraphicState::setProjectionMatrix(glm::mat4 projection)
// {
// 	this->projection = projection;

// 	buffers[(int)CameraBuffer].setSubData(glm::value_ptr(projection), 0, 64);
// }

// void GraphicState::setViewMatrix(glm::mat4 view)
// {
// 	this->view = view;

// 	buffers[(int)CameraBuffer].setSubData(glm::value_ptr(view), 64, 64);
// }

// void GraphicState::setCameraPosition(glm::vec3 position)
// {
// 	this->cameraPos = position;

// 	buffers[(int)CameraBuffer].setSubData(glm::value_ptr(position), 128, 16);
// }

// void GraphicState::setLightProjectionMatrix(glm::mat4 projection, int index)
// {
// 	this->lightProjection[index] = projection;

// 	buffers[(int)ShadowBuffer].setSubData(glm::value_ptr(projection), 64 * index, 64);
// }

// void GraphicState::setLightViewMatrix(glm::mat4 view, int index)
// {
// 	this->lightView[index] = view;

// 	buffers[(int)ShadowBuffer].setSubData(glm::value_ptr(view), 320 + 64 * index, 64);
// }

// void GraphicState::setCascadeEnd(float cascadeEnd, int index)
// {
// 	this->cascadeEnd[index] = cascadeEnd;

// 	buffers[(int)ShadowBuffer].setSubData(&cascadeEnd, 640 + 16 * index, 16);
// }

// void GraphicState::setFarPlane(float farPlane)
// {
// 	this->farPlane = farPlane;

// 	buffers[(int)ShadowBuffer].setSubData(&farPlane, 720, 16);
// }

// void GraphicState::setDirLightDirection(glm::vec3 direction)
// {
// 	this->dirLightDirection = direction;

// 	buffers[(int)DirectionalLightBuffer].setSubData(glm::value_ptr(direction), 0, 16);
// }

// void GraphicState::setDirLightAmbient(glm::vec3 ambient)
// {
// 	this->dirLightAmbient = ambient;

// 	buffers[(int)DirectionalLightBuffer].setSubData(glm::value_ptr(ambient), 16, 16);
// }

// void GraphicState::setDirLightDiffuse(glm::vec3 diffuse)
// {
// 	this->dirLightDiffuse = diffuse;

// 	buffers[(int)DirectionalLightBuffer].setSubData(glm::value_ptr(diffuse), 32, 16);
// }

// void GraphicState::setDirLightSpecular(glm::vec3 specular)
// {
// 	this->dirLightSpecular = specular;

// 	buffers[(int)DirectionalLightBuffer].setSubData(glm::value_ptr(specular), 48, 16);
// }

// void GraphicState::setSpotLightPosition(glm::vec3 position)
// {
// 	this->spotLightPosition = position;

// 	buffers[(int)SpotLightBuffer].setSubData(glm::value_ptr(position), 0, 16);
// }

// void GraphicState::setSpotLightDirection(glm::vec3 direction)
// {
// 	this->spotLightDirection = direction;

// 	buffers[(int)SpotLightBuffer].setSubData(glm::value_ptr(direction), 16, 16);
// }

// void GraphicState::setSpotLightAmbient(glm::vec3 ambient)
// {
// 	this->spotLightAmbient = ambient;

// 	buffers[(int)SpotLightBuffer].setSubData(glm::value_ptr(ambient), 32, 16);
// }

// void GraphicState::setSpotLightDiffuse(glm::vec3 diffuse)
// {
// 	this->spotLightDiffuse = diffuse;

// 	buffers[(int)SpotLightBuffer].setSubData(glm::value_ptr(diffuse), 48, 16);
// }

// void GraphicState::setSpotLightSpecular(glm::vec3 specular)
// {
// 	this->spotLightSpecular = specular;

// 	buffers[(int)SpotLightBuffer].setSubData(glm::value_ptr(specular), 64, 16);
// }

// void GraphicState::setSpotLightConstant(float constant)
// {
// 	this->spotLightConstant = constant;

// 	buffers[(int)SpotLightBuffer].setSubData(&constant, 80, 4);
// }

// void GraphicState::setSpotLightLinear(float linear)
// {
// 	this->spotLightLinear = linear;

// 	buffers[(int)SpotLightBuffer].setSubData(&linear, 84, 4);
// }

// void GraphicState::setSpotLightQuadratic(float quadratic)
// {
// 	this->spotLightQuadratic = quadratic;

// 	buffers[(int)SpotLightBuffer].setSubData(&quadratic, 88, 4);
// }

// void GraphicState::setSpotLightCutoff(float cutoff)
// {
// 	this->spotLightCutoff = cutoff;

// 	buffers[(int)SpotLightBuffer].setSubData(&cutoff, 92, 4);
// }

// void GraphicState::setSpotLightOuterCutoff(float cutoff)
// {
// 	this->spotLightOuterCutoff = cutoff;

// 	buffers[(int)SpotLightBuffer].setSubData(&cutoff, 96, 4);
// }

// void GraphicState::setPointLightPosition(glm::vec3 position)
// {
// 	this->pointLightPosition = position;

// 	buffers[(int)PointLightBuffer].setSubData(glm::value_ptr(position), 0, 16);
// }

// void GraphicState::setPointLightAmbient(glm::vec3 ambient)
// {
// 	this->pointLightAmbient = ambient;

// 	buffers[(int)PointLightBuffer].setSubData(glm::value_ptr(ambient), 16, 16);
// }

// void GraphicState::setPointLightDiffuse(glm::vec3 diffuse)
// {
// 	this->pointLightDiffuse = diffuse;

// 	buffers[(int)PointLightBuffer].setSubData(glm::value_ptr(diffuse), 32, 16);
// }

// void GraphicState::setPointLightSpecular(glm::vec3 specular)
// {
// 	this->pointLightSpecular = specular;

// 	buffers[(int)PointLightBuffer].setSubData(glm::value_ptr(specular), 48, 16);
// }

// void GraphicState::setPointLightConstant(float constant)
// {
// 	this->pointLightConstant = constant;

// 	buffers[(int)PointLightBuffer].setSubData(&constant, 64, 4);
// }

// void GraphicState::setPointLightLinear(float linear)
// {
// 	this->pointLightLinear = linear;

// 	buffers[(int)PointLightBuffer].setSubData(&linear, 68, 4);
// }

// void GraphicState::setPointLightQuadratic(float quadratic)
// {
// 	this->pointLightQuadratic = quadratic;

// 	buffers[(int)PointLightBuffer].setSubData(&quadratic, 72, 4);
// }




// // getters

// glm::mat4 GraphicState::getProjectionMatrix()
// {
// 	return projection;
// }

// glm::mat4 GraphicState::getViewMatrix()
// {
// 	return view;
// }

// glm::vec3 GraphicState::getCameraPosition()
// {
// 	return cameraPos;
// }

// glm::mat4 GraphicState::getLightProjectionMatrix(int index)
// {
// 	return lightProjection[index];
// }

// glm::mat4 GraphicState::getLightViewMatrix(int index)
// {
// 	return lightView[index];
// }

// float GraphicState::getCascadeEnd(int index)
// {
// 	return cascadeEnd[index];
// }

// float GraphicState::getFarPlane()
// {
// 	return farPlane;
// }

// glm::vec3 GraphicState::getDirLightDirection()
// {
// 	return dirLightDirection;
// }

// glm::vec3 GraphicState::getDirLightAmbient()
// {
// 	return dirLightAmbient;
// }

// glm::vec3 GraphicState::getDirLightDiffuse()
// {
// 	return dirLightDiffuse;
// }

// glm::vec3 GraphicState::getDirLightSpecular()
// {
// 	return dirLightSpecular;
// }

// glm::vec3 GraphicState::getSpotLightDirection()
// {
// 	return spotLightDirection;
// }

// glm::vec3 GraphicState::getSpotLightPosition()
// {
// 	return spotLightPosition;
// }

// glm::vec3 GraphicState::getSpotLightAmbient()
// {
// 	return spotLightAmbient;
// }

// glm::vec3 GraphicState::getSpotLightDiffuse()
// {
// 	return spotLightDiffuse;
// }

// glm::vec3 GraphicState::getSpotLightSpecular()
// {
// 	return spotLightSpecular;
// }

// glm::vec3 GraphicState::getPointLightPosition()
// {
// 	return pointLightPosition;
// }

// glm::vec3 GraphicState::getPointLightAmbient()
// {
// 	return pointLightAmbient;
// }

// glm::vec3 GraphicState::getPointLightDiffuse()
// {
// 	return pointLightDiffuse;
// }

// glm::vec3 GraphicState::getPointLightSpecular()
// {
// 	return pointLightSpecular;
// }

// float GraphicState::getSpotLightConstant()
// {
// 	return spotLightConstant;
// }

// float GraphicState::getSpotLightLinear()
// {
// 	return spotLightLinear;
// }

// float GraphicState::getSpotLightQuadratic()
// {
// 	return spotLightQuadratic;
// }

// float GraphicState::getSpotLightCutoff()
// {
// 	return spotLightCutoff;
// }

// float GraphicState::getSpotLightOuterCutoff()
// {
// 	return spotLightOuterCutoff;
// }

// float GraphicState::getPointLightConstant()
// {
// 	return pointLightConstant;
// }

// float GraphicState::getPointLightLinear()
// {
// 	return pointLightLinear;
// }

// float GraphicState::getPointLightQuadratic()
// {
// 	return pointLightQuadratic;
// }