#include <iostream>

#include "../../include/core/LoadInternal.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Entity.h"
#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Texture3D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Material.h"
#include "../../include/core/Mesh.h"
#include "../../include/core/Font.h"

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
#include "../../include/components/SphereCollider.h"
#include "../../include/components/MeshCollider.h"
#include "../../include/components/Boids.h"
#include "../../include/components/Cloth.h"
#include "../../include/components/Fluid.h"
#include "../../include/components/Solid.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"
#include "../../include/systems/BoidsSystem.h"
#include "../../include/systems/ClothSystem.h"
#include "../../include/systems/FluidSystem.h"
#include "../../include/systems/SolidSystem.h"

using namespace PhysicsEngine;

Asset* PhysicsEngine::loadInternalAsset(std::vector<char> data, int* index)  
{
	int type = *reinterpret_cast<int*>(&data[0]); //maybe should add classification to assets? Just for consistency with components, entities, and systems?

	if(type == 0){
		std::cout << "a shader allocator count: " << getAllocator<Shader>().getCount() << std::endl;
		*index = (int)getAllocator<Shader>().getCount();
		return create<Shader>(data);
	}
	else if(type == 1){
		*index = (int)getAllocator<Texture2D>().getCount();
		return create<Texture2D>(data);
	}
	else if(type == 2){
		*index = (int)getAllocator<Texture3D>().getCount();
		return create<Texture3D>(data);
	}
	else if(type == 3){
		*index = (int)getAllocator<Cubemap>().getCount();
		return create<Cubemap>(data);
	}
	else if(type == 4){
		*index = (int)getAllocator<Material>().getCount();
		return create<Material>(data);
	}
	else if(type == 5){
		*index = (int)getAllocator<Mesh>().getCount();
		return create<Mesh>(data);
	}
	else if(type == 6){
		*index = (int)getAllocator<Font>().getCount();
		return create<Font>(data);
	}
	else{
		std::cout << "Error: Invalid asset type (" << type << ") when trying to load internal asset" << std::endl;
		return NULL;
	}
}

Entity* PhysicsEngine::loadInternalEntity(std::vector<char> data, int* index)
{
	*index = (int)getAllocator<Entity>().getCount();
	return create<Entity>(data);
}

Component* PhysicsEngine::loadInternalComponent(std::vector<char> data, int* index, int* instanceType)
{
	//char classification = *reinterpret_cast<char*>(&data[0]);
	int type = *reinterpret_cast<int*>(&data[sizeof(char)]);

	if(type == 0){
		*index = (int)getAllocator<Transform>().getCount();
		*instanceType = (int)Component::getInstanceType<Transform>();
		return create<Transform>(data);
	}
	else if(type == 1){
		*index = (int)getAllocator<Rigidbody>().getCount();
		*instanceType = (int)Component::getInstanceType<Rigidbody>();
		return create<Rigidbody>(data);
	}
	else if(type == 2){
		*index = (int)getAllocator<Camera>().getCount();
		*instanceType = (int)Component::getInstanceType<Camera>();
		return create<Camera>(data);
	}
	else if(type == 3){
		*index = (int)getAllocator<MeshRenderer>().getCount();
		*instanceType = (int)Component::getInstanceType<MeshRenderer>();
		return create<MeshRenderer>(data);
	}
	else if(type == 4){
		*index = (int)getAllocator<LineRenderer>().getCount();
		*instanceType = (int)Component::getInstanceType<LineRenderer>();
		return create<LineRenderer>(data);
	}
	else if(type == 5){
		*index = (int)getAllocator<DirectionalLight>().getCount();
		*instanceType = (int)Component::getInstanceType<DirectionalLight>();
		return create<DirectionalLight>(data);
	}
	else if(type == 6){
		*index = (int)getAllocator<SpotLight>().getCount();
		*instanceType = (int)Component::getInstanceType<SpotLight>();
		return create<SpotLight>(data);
	}
	else if(type == 7){
		*index = (int)getAllocator<PointLight>().getCount();
		*instanceType = (int)Component::getInstanceType<PointLight>();
		return create<PointLight>(data);
	}
	else if(type == 8){
		*index = (int)getAllocator<BoxCollider>().getCount();
		*instanceType = (int)Component::getInstanceType<BoxCollider>();
		return create<BoxCollider>(data);
	}
	else if(type == 9){
		*index = (int)getAllocator<SphereCollider>().getCount();
		*instanceType = (int)Component::getInstanceType<SphereCollider>();
		return create<SphereCollider>(data);
	}
	else if(type == 15){
		*index = (int)getAllocator<MeshCollider>().getCount();
		*instanceType = (int)Component::getInstanceType<MeshCollider>();
		return create<MeshCollider>(data);
	}
	else if(type == 10){
		*index = (int)getAllocator<CapsuleCollider>().getCount();
		*instanceType = (int)Component::getInstanceType<CapsuleCollider>();
		return create<CapsuleCollider>(data);
	}
	else if(type == 11){
		*index = (int)getAllocator<Boids>().getCount();
		*instanceType = (int)Component::getInstanceType<Boids>();
		return create<Boids>(data);
	}
	else if(type == 12){
		*index = (int)getAllocator<Cloth>().getCount();
		*instanceType = (int)Component::getInstanceType<Cloth>();
		return create<Cloth>(data);
	}
	else if(type == 13){
		*index = (int)getAllocator<Fluid>().getCount();
		*instanceType = (int)Component::getInstanceType<Fluid>();
		return create<Fluid>(data);
	}
	else if(type == 14){
		*index = (int)getAllocator<Solid>().getCount();
		*instanceType = (int)Component::getInstanceType<Solid>();
		return create<Solid>(data);
	}
	else{
		std::cout << "Error: Invalid component type (" << type << ") when trying to load internal component" << std::endl;
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(std::vector<char> data, int* index)
{
	//char classification = *reinterpret_cast<char*>(&data[0]);
	int type = *reinterpret_cast<int*>(&data[sizeof(char)]);

	if(type == 0){
		*index = (int)getAllocator<RenderSystem>().getCount();
		return create<RenderSystem>(data);
	}
	else if(type == 1){
		*index = (int)getAllocator<PhysicsSystem>().getCount();
		return create<PhysicsSystem>(data);
	}
	else if(type == 2){
		*index = (int)getAllocator<CleanUpSystem>().getCount();
		return create<CleanUpSystem>(data);
	}
	else if(type == 3){
		*index = (int)getAllocator<DebugSystem>().getCount();
		return create<DebugSystem>(data);
	}
	else if(type == 4){
		*index = (int)getAllocator<BoidsSystem>().getCount();
		return create<BoidsSystem>(data);
	}
	else if(type == 5){
		*index = (int)getAllocator<ClothSystem>().getCount();
		return create<ClothSystem>(data);
	}
	else if(type == 6){
		*index = (int)getAllocator<FluidSystem>().getCount();
		return create<FluidSystem>(data);
	}
	else if(type == 7){
		*index = (int)getAllocator<SolidSystem>().getCount();
		return create<SolidSystem>(data);
	}
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load internal system" << std::endl;
		return NULL;
	}
}