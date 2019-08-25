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
#include "../../include/components/Light.h"
#include "../../include/components/BoxCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/MeshCollider.h"
// #include "../../include/components/Cloth.h"
// #include "../../include/components/Fluid.h"
// #include "../../include/components/Solid.h"
// #include "../../include/components/Boids.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"
// #include "../../include/systems/BoidsSystem.h"
// #include "../../include/systems/ClothSystem.h"
// #include "../../include/systems/FluidSystem.h"
// #include "../../include/systems/SolidSystem.h"

using namespace PhysicsEngine;

Asset* PhysicsEngine::loadInternalAsset(std::vector<char> data, int type, int* index)  
{
	if(type == 0){
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

Component* PhysicsEngine::loadInternalComponent(std::vector<char> data, int type, int* index)
{
	if(type == 0){
		*index = (int)getAllocator<Transform>().getCount();
		return create<Transform>(data);
	}
	else if(type == 1){
		*index = (int)getAllocator<Rigidbody>().getCount();
		return create<Rigidbody>(data);
	}
	else if(type == 2){
		*index = (int)getAllocator<Camera>().getCount();
		return create<Camera>(data);
	}
	else if(type == 3){
		*index = (int)getAllocator<MeshRenderer>().getCount();
		return create<MeshRenderer>(data);
	}
	else if(type == 4){
		*index = (int)getAllocator<LineRenderer>().getCount();
		return create<LineRenderer>(data);
	}
	else if(type == 5){
		*index = (int)getAllocator<Light>().getCount();
		return create<Light>(data);
	}
	else if(type == 8){
		*index = (int)getAllocator<BoxCollider>().getCount();
		return create<BoxCollider>(data);
	}
	else if(type == 9){
		*index = (int)getAllocator<SphereCollider>().getCount();
		return create<SphereCollider>(data);
	}
	else if(type == 15){
		*index = (int)getAllocator<MeshCollider>().getCount();
		return create<MeshCollider>(data);
	}
	else if(type == 10){
		*index = (int)getAllocator<CapsuleCollider>().getCount();
		return create<CapsuleCollider>(data);
	}
	// else if(type == 11){
	// 	*index = (int)getAllocator<Boids>().getCount();
	// 	return create<Boids>(data);
	// }
	// else if(type == 12){
	// 	*index = (int)getAllocator<Cloth>().getCount();
	// 	return create<Cloth>(data);
	// }
	// else if(type == 13){
	// 	*index = (int)getAllocator<Fluid>().getCount();
	// 	return create<Fluid>(data);
	// }
	// else if(type == 14){
	// 	*index = (int)getAllocator<Solid>().getCount();
	// 	return create<Solid>(data);
	// }
	else{
		std::cout << "Error: Invalid component type (" << type << ") when trying to load internal component" << std::endl;
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(std::vector<char> data, int type, int* index)
{
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
	// else if(type == 4){
	// 	*index = (int)getAllocator<BoidsSystem>().getCount();
	// 	return create<BoidsSystem>(data);
	// }
	// else if(type == 5){
	// 	*index = (int)getAllocator<ClothSystem>().getCount();
	// 	return create<ClothSystem>(data);
	// }
	// else if(type == 6){
	// 	*index = (int)getAllocator<FluidSystem>().getCount();
	// 	return create<FluidSystem>(data);
	// }
	// else if(type == 7){
	// 	*index = (int)getAllocator<SolidSystem>().getCount();
	// 	return create<SolidSystem>(data);
	// }
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load internal system" << std::endl;
		return NULL;
	}
}

Entity* PhysicsEngine::destroyInternalEntity(int index)
{
	return destroy<Entity>(index);
}

Component* PhysicsEngine::destroyInternalComponent(int type, int index)
{
	if(type == ComponentType<Transform>::type){
		return destroy<Transform>(index);
	}
	else if(type == ComponentType<Rigidbody>::type){
		return destroy<Rigidbody>(index);
	}
	else if(type == ComponentType<Camera>::type){
		return destroy<Camera>(index);
	}
	else if(type == ComponentType<MeshRenderer>::type){
		return destroy<MeshRenderer>(index);
	}
	else if(type == ComponentType<LineRenderer>::type){
		return destroy<LineRenderer>(index);
	}
	else if(type == ComponentType<Light>::type){
		return destroy<Light>(index);
	}
	else if(type == ComponentType<BoxCollider>::type){
		return destroy<BoxCollider>(index);
	}
	else if(type == ComponentType<SphereCollider>::type){
		return destroy<SphereCollider>(index);
	}
	else if(type == ComponentType<MeshCollider>::type){
		return destroy<MeshCollider>(index);
	}
	else if(type == ComponentType<CapsuleCollider>::type){
		return destroy<CapsuleCollider>(index);
	}
	// else if(type == ComponentType<Boids>::type){
	// 	return destroy<Boids>(index);
	// }
	// else if(type == ComponentType<Cloth>::type){
	// 	return destroy<Cloth>(index);
	// }
	// else if(type == ComponentType<Fluid>::type){
	// 	return destroy<Fluid>(index);
	// }
	// else if(type == ComponentType<Solid>::type){
	// 	return destroy<Solid>(index);
	// }
	else{
		std::cout << "Error: Invalid component instance type (" << type << ") when trying to destroy internal component" << std::endl;
		return NULL;
	}
}