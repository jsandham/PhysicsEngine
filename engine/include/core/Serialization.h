#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include <iostream>
#include <fstream>

#include "Entity.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Material.h"
#include "Mesh.h"
#include "GMesh.h"

#include "../components/Transform.h"
#include "../components/Rigidbody.h"
#include "../components/Camera.h"
#include "../components/DirectionalLight.h"
#include "../components/PointLight.h"
#include "../components/SpotLight.h"
#include "../components/MeshRenderer.h"
#include "../components/LineRenderer.h"
#include "../components/Collider.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/CapsuleCollider.h"

namespace PhysicsEngine
{
	bool deserializeShader(Shader* shader, std::ifstream& file)
{
	if(!file.is_open()){
		std::cout << "Error: Input file stream must be open before calling deserialize on shader" << std::endl;
		return false;
	}

	ShaderHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(ShaderHeader));

	bool error = false;
	error |= header.vertexShaderSize <= 0 || header.vertexShaderSize > 10000;
	error |= header.geometryShaderSize < 0 || header.geometryShaderSize > 10000;
	error |= header.fragmentShaderSize <= 0 || header.fragmentShaderSize > 10000;

	if(error){
		std::cout << "Error: vertex, geometry, and/or fragment shader must not have invalid size" << std::endl;
		return false;
	}

	std::cout << "asset id: " << header.shaderId.toString() << std::endl;
	std::cout << "vertex shader size: " << header.vertexShaderSize << std::endl;
	std::cout << "fragment shader size: " << header.fragmentShaderSize << std::endl;
	std::cout << "geometry shader size: " << header.geometryShaderSize << std::endl;

	std::vector<char> vertexShaderData(header.vertexShaderSize);
	file.read(&vertexShaderData[0], header.vertexShaderSize * sizeof(char));
	std::string vertexShader = std::string(vertexShaderData.begin(), vertexShaderData.end());

	std::string geometryShader = "";
	if(header.geometryShaderSize > 0){
		std::vector<char> geometryShaderData(header.geometryShaderSize);
		file.read(&geometryShaderData[0], header.geometryShaderSize * sizeof(char));
		geometryShader = std::string(geometryShaderData.begin(), geometryShaderData.end());
	}

	std::vector<char> fragmentShaderData(header.fragmentShaderSize);
	file.read(&fragmentShaderData[0], header.fragmentShaderSize * sizeof(char));
	std::string fragmentShader = std::string(fragmentShaderData.begin(), fragmentShaderData.end());

	shader->assetId = header.shaderId;
	shader->vertexShader = vertexShader;
	shader->fragmentShader = fragmentShader;
	shader->geometryShader = geometryShader;

	// std::cout << "asset id: " << shader->assetId.toString() << " vertex shader: " << shader->vertexShader << " fragment shader: " << shader->fragmentShader << " geometry shader: " << shader->geometryShader << std::endl;

	return true;
}

bool deserializeTexture2D(Texture2D* texture, std::ifstream& file)
{
	if(!file.is_open()){
		std::cout << "Error: Input file stream must be open before calling deserialize on texture" << std::endl;
		return false;
	}

	Texture2DHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(Texture2DHeader));

	std::cout << "width: " << header.width << std::endl;
	std::cout << "height: " << header.height << std::endl;
	std::cout << "dimension: " << header.dimension << std::endl;
	std::cout << "format: " << header.format << std::endl;

	std::vector<unsigned char> data(header.textureSize);
	file.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(unsigned char));

	texture->assetId = header.textureId;
	texture->setRawTextureData(data, header.width, header.height, static_cast<TextureFormat>(header.format));

	return true;
}

bool deserializeMaterial(Material* material, std::ifstream& file)
{
	if(!file.is_open()){
		std::cout << "Error: Input file stream must be open before calling deserialize on material" << std::endl;
		return false;
	}

	MaterialHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(MaterialHeader));

	material->assetId = header.assetId;
	material->shaderId = header.shaderId;
	material->textureId = header.textureId;
	material->normalMapId = header.normalMapId;
	material->specularMapId = header.specularMapId;

	material->shininess = header.shininess;
	material->ambient = header.ambient;
	material->diffuse = header.diffuse;
	material->specular = header.specular;
	material->color = header.color;

	return true;
}

bool deserializeMesh(Mesh* mesh, std::ifstream& file)
{
	if(!file.is_open()){
		std::cout << "Error: Input file stream must be open before calling deserialize on mesh" << std::endl;
		return false;
	}

	MeshHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(MeshHeader));

	std::vector<float> vertices(header.verticesSize);
	std::vector<float> normals(header.normalsSize);
	std::vector<float> texCoords(header.texCoordsSize);

	file.read(reinterpret_cast<char*>(&vertices[0]), header.verticesSize * sizeof(float));
	file.read(reinterpret_cast<char*>(&normals[0]), header.normalsSize * sizeof(float));
	file.read(reinterpret_cast<char*>(&texCoords[0]), header.texCoordsSize * sizeof(float));

	mesh->assetId = header.meshId;
	mesh->vertices = vertices;
	mesh->normals = normals;
	mesh->texCoords = texCoords;

	return true;
}

bool deserializeGMesh(GMesh* gmesh, std::ifstream& file)
{
	if(!file.is_open()){
		std::cout << "Error: Input file stream must be open before calling deserialize on gmesh" << std::endl;
		return false;
	}

	GMeshHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(GMeshHeader));

	std::vector<float> vertices(header.verticesSize);
	std::vector<int> connect(header.connectSize);
	std::vector<int> bconnect(header.bconnectSize);	
	std::vector<int> groups(header.groupsSize);

	file.read(reinterpret_cast<char*>(&vertices[0]), header.verticesSize * sizeof(float));
	file.read(reinterpret_cast<char*>(&connect[0]), header.connectSize * sizeof(int));
	file.read(reinterpret_cast<char*>(&bconnect[0]), header.bconnectSize * sizeof(int));
	file.read(reinterpret_cast<char*>(&groups[0]), header.groupsSize * sizeof(int));

	gmesh->assetId = header.gmeshId;
	gmesh->dim = header.dim;
	gmesh->ng = header.ng;
    gmesh->n = header.n;
    gmesh->nte = header.nte;
    gmesh->ne = header.ne;
    gmesh->ne_b = header.ne_b;
    gmesh->npe = header.npe;
    gmesh->npe_b = header.npe_b;
    gmesh->type = header.type;
    gmesh->type_b = header.type_b;

    gmesh->vertices = vertices;
    gmesh->connect = connect;
    gmesh->bconnect = bconnect;
    gmesh->groups = groups;

	return true;
}

	// bool deserializeEntity(Entity* entity, std::ifstream& file);
	// bool deserializeTransform(Transform* transform, std::ifstream& file);
	// bool deserializeRigidbody(Rigidbody* rigidbody, std::ifstream& file);
	// bool deserializeCamera(Camera* camera, std::ifstream& file);
	// bool deserializeMeshRenderer(MeshRenderer* meshRenderer, std::ifstream& file);
	// bool deserializeLineRenderer(LineRenderer* lineRenderer, std::ifstream& file);
	// bool deserializeDirectionalLight(DirectionalLight* directionalLight, std::ifstream& file);
	// bool deserializeSpotLight(SpotLight* spotLight, std::ifstream& file);
	// bool deserializePointLight(PointLight* pointLight, std::ifstream& file);
	// bool deserializeBoxCollider(BoxCollider* boxCollider, std::ifstream& file);
	// bool deserializeSphereCollider(SphereCollider* sphereCollider, std::ifstream& file);
	// bool deserializeCapsuleCollider(CapsuleCollider* capsuleCollider, std::ifstream& file);


	// bool serializeShader(std::string shaderFilePath, std::ofstream& file)
	// {

	// }
}


#endif