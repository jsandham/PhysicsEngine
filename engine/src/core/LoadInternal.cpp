#include "../../include/core/LoadInternal.h"

#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Material.h"
#include "../../include/core/Mesh.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Rigidbody.h"
#include "../../include/components/Camera.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/LineRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/BoxCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/CapsuleCollider.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"

using namespace PhysicsEngine;

Asset* loadInternalAsset(unsigned char* data)
{
	int type = *reinterpret_cast<int*>(data);

	if(type == 0){
		return new Shader(data);
	}
	else if(type == 1){
		return new Texture2D(data);
	}
	else if(type == 2){
		return new Material(data);
	}
	else if(type == 3){
		return new Mesh(data);
	}
	else{
		std::cout << "Error: Invalid asset type (" << type << ") when trying to load internal asset" << std::endl;
		return NULL;
	}
}

Entity* loadInternalEntity(unsigned char* data)
{
	return new Entity(data);
}

Component* loadInternalComponent(unsigned char* data)
{
	int type = *reinterpret_cast<int*>(data);

	if(type == 0){
		return new Transform(data);
	}
	else if(type == 1){
		return new Rigidbody(data);
	}
	else if(type == 2){
		return new Camera(data);
	}
	else if(type == 3){
		return new MeshRenderer(data);
	}
	else if(type == 4){
		return new LineRenderer(data);
	}
	else if(type == 5){
		return new DirectionalLight(data);
	}
	else if(type == 6){
		return new SpotLight(data);
	}
	else if(type == 7){
		return new PointLight(data);
	}
	else if(type == 8){
		return new BoxCollider(data);
	}
	else if(type == 9){
		return new SphereCollider(data);
	}
	else if(type == 10){
		return new CapsuleCollider(data);
	}
	else{
		std::cout << "Error: Invalid component type (" << type << ") when trying to load internal component" << std::endl;
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(unsigned char* data)
{
	int type = *reinterpret_cast<int*>(data);

	if(type == 0){
		return new RenderSystem(data);
	}
	else if(type == 1){
		return new PhysicsSystem(data);
	}
	else if(type == 2){
		return new CleanUpSystem(data);
	}
	else if(type == 3){
		return new DebugSystem(data);
	}
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load internal system" << std::endl;
		return NULL;
	}
}

