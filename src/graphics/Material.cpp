#include "Material.h"

using namespace PhysicsEngine;


Material::Material(Shader* shader)
{
	this->shader = shader;

	this->shininess = 1.0f;
	this->color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	this->ambient = glm::vec3(0.1f, 0.1f, 0.1f);
	this->diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	this->specular = glm::vec3(1.0f, 1.0f, 1.0f);

	uniforms = new ShaderUniformState(shader);

	textures.resize(7);
}

Material::Material(const Material &material)
{
	shader = material.shader;
	textures = material.textures;

	shininess = material.shininess;
	color = material.color;
	ambient = material.ambient;
	diffuse = material.diffuse;
	specular = material.specular;

	uniforms = new ShaderUniformState(material.shader);

	mainTexture = material.mainTexture;
	normalMap = material.normalMap;
	diffuseMap = material.diffuseMap;
	specularMap = material.specularMap;
	albedoMap = material.albedoMap;
	glossMap = material.glossMap;
	cubemap = material.cubemap;
}

Material& Material::operator=(const Material &material)
{
	if (this != &material){

	}

	return *this;
}

Material::~Material()
{
	delete uniforms;
}

void Material::bind(GraphicState& state)
{
	shader->bind();

	shader->setFloat("material.shininess", shininess);
	shader->setVec4("color", color);
	shader->setVec3("material.ambient", ambient);
	shader->setVec3("material.diffuse", diffuse);
	shader->setVec3("material.specular", specular);

	for (unsigned int i = 0; i < textures.size(); i++){
		if (textures[i] != NULL){
			textures[i]->active(i);
			textures[i]->bind();
		}
	}

	for (unsigned int i = 0; i < state.cascadeTexture2D.size(); i++){
		Texture2D* texture = state.cascadeTexture2D[i];
		if (texture != NULL){
			texture->active(i + 7);
			texture->bind();

			std::string name = "shadowMap[" + std::to_string(i) + "]";
			shader->setInt(name, i + 7);
		}
	}

	if (state.shadowTexture2D != NULL){
		state.shadowTexture2D->active(7);
		state.shadowTexture2D->bind();

		shader->setInt("shadowMap[0]", 7);
	}

	if (state.shadowCubemap != NULL){
		state.shadowCubemap->active(7);
		state.shadowCubemap->bind();

		shader->setInt("cubeShadowMap", 7);
	}

	uniforms->setUniforms();
}

void Material::unbind()
{
	shader->unbind();

	for (unsigned int i = 0; i < textures.size(); i++){
		if (textures[i] != NULL){
			textures[i]->active(i);
			textures[i]->unbind();
		}
	}
}

void Material::setShininess(float shininess)
{
	this->shininess = shininess;
}

void Material::setColor(glm::vec4 &color)
{
	this->color = color;

	//uniforms->setVec4("color", color);
}

void Material::setAmbient(glm::vec3 &ambient)
{
	this->ambient = ambient;

	//uniforms->setVec3("ambient", ambient);
}

void Material::setDiffuse(glm::vec3 &diffuse)
{
	this->diffuse = diffuse;

	//uniforms->setVec3("diffuse", diffuse);
}

void Material::setSpecular(glm::vec3 &specular)
{
	this->specular = specular;

	//uniforms->setVec3("specular", specular);
}

void Material::setMainTexture(Texture2D *texture)
{
	textures[MAINTEXTURE] = texture;
	mainTexture = texture;
	uniforms->setInt("mainTexture", MAINTEXTURE);
}

void Material::setNormalMap(Texture2D *texture)
{
	textures[NORMAL] = texture;
	normalMap = texture;
	uniforms->setInt("normalMap", NORMAL);
}

void Material::setDiffuseMap(Texture2D *texture)
{
	textures[DIFFUSE] = texture;
	diffuseMap = texture;
	uniforms->setInt("diffuseMap", DIFFUSE);
}

void Material::setSpecularMap(Texture2D *texture)
{
	textures[SPECULAR] = texture;
	specularMap = texture;
	uniforms->setInt("specularMap", SPECULAR);
}

void Material::setAlbedoMap(Texture2D *texture)
{
	textures[ALBEDO] = texture;
	albedoMap = texture;
	uniforms->setInt("albedoMap", ALBEDO);
}

void Material::setGlossMap(Texture2D *texture)
{
	textures[GLOSS] = texture;
	glossMap = texture;
	uniforms->setInt("glossMap", GLOSS);
}

//void Material::setCubemap(Cubemap *texture)
//{
//	textures[CUBEMAP] = texture;
//	cubemap = texture;
//	uniforms->setInt("cubemap", CUBEMAP);
//}

void Material::setBool(std::string name, bool value)
{
	uniforms->setBool(name, value);
}

void Material::setInt(std::string name, int value)
{
	uniforms->setInt(name, value);
}

void Material::setFloat(std::string name, float value)
{
	uniforms->setFloat(name, value);
}

void Material::setVec2(std::string name, glm::vec2 &vec)
{
	uniforms->setVec2(name, vec);
}

void Material::setVec3(std::string name, glm::vec3 &vec)
{
	uniforms->setVec3(name, vec);
}

void Material::setVec4(std::string name, glm::vec4 &vec)
{
	uniforms->setVec4(name, vec);
}

void Material::setMat2(std::string name, glm::mat2 &mat)
{
	uniforms->setMat2(name, mat);
}

void Material::setMat3(std::string name, glm::mat3 &mat)
{
	uniforms->setMat3(name, mat);
}

void Material::setMat4(std::string name, glm::mat4 &mat)
{
	uniforms->setMat4(name, mat);
}

float Material::getShininess()
{
	return shininess;
}

glm::vec4& Material::getColor()
{
	return color;
}

glm::vec3& Material::getAmbient()
{
	return ambient;
}

glm::vec3& Material::getDiffuse()
{
	return diffuse;
}

glm::vec3& Material::getSpecular()
{
	return specular;
}

Texture2D* Material::getMainTexture()
{
	return mainTexture;
}

Texture2D* Material::getNormalMap()
{
	return normalMap;
}

Texture2D* Material::getDiffuseMap()
{
	return diffuseMap;
}

Texture2D* Material::getSpecularMap()
{
	return specularMap;
}

Texture2D* Material::getAlbedoMap()
{
	return albedoMap;
}

Texture2D* Material::getGlossMap()
{
	return glossMap;
}

Cubemap* Material::getCubemap()
{
	return cubemap;
}

Shader* Material::getShader()
{
	return shader;
}