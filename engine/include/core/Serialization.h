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
	bool deserializeShader(Shader* shader, std::ifstream& file);
	bool deserializeTexture2D(Texture2D* texture, std::ifstream& file);
	bool deserializeMaterial(Material* material, std::ifstream& file);
	bool deserializeMesh(Mesh* mesh, std::ifstream& file);
	bool deserializeGMesh(GMesh* gmesh, std::ifstream& file);

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