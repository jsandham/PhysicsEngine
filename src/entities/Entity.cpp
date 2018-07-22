#include "Entity.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	
}

Entity::~Entity()
{

}
























//int Entity::entityIdCount = 0;
//std::vector<Entity> Entity::entities(10, Entity());
//
//Entity::Entity()
//{
//	std::cout << "entity constructor called for entity with id " << entityId << std::endl;
//}
//
//Entity::~Entity()
//{
//	std::cout << "entity destructor called for entity with id " << entityId << std::endl;
//}
//
//void Entity::Destroy(Entity *entity)
//{
//	
//}
//
//Entity* Entity::Instantiate()
//{
//	if (entityIdCount == 10){
//		std::cout << "Entity: reached maximum number of entities allow" << std::endl;
//		return NULL;
//	}
//
//	Entity *entity = &Entity::entities[entityIdCount];
//
//	entity->entityId = entityIdCount;
//	entity->name = "Entity";
//	entity->AddComponent<Transform>();
//
//	entityIdCount++;
//
//	std::cout << "entity instantiate called for entity with id " << entity->entityId << std::endl;
//	
//	return entity;
//}


























//std::map<std::type_index, int> Entity::indiciesToComponentMaps = Entity::createIndiciesToComponentMaps();
//std::vector<std::map<int, std::vector<Component*> >> Entity::components = Entity::createComponentMaps();
//
//
//int Entity::entityIdCount = 0;
//std::map<int, Entity*> Entity::entities;
//
//
//Entity::Entity()
//{
//	entityId = entityIdCount;
//	entityIdCount++;
//
//	std::cout << "entity constructor called for entity with id " << entityId << std::endl;
//
//	name = "Entity";
//
//	AddComponent<Transform>();
//}
//
//Entity::~Entity()
//{
//	std::cout << "entity destructor called for entity with id " << entityId << std::endl;
//}
//
//
//void Entity::Destroy(Entity *entity)
//{
//	std::cout << "entity destroy called on entity with id: " << entity->entityId << std::endl;
//
//	std::map<int, Entity*>::iterator it = entities.find(entity->entityId);
//	if (it != entities.end()){
//		for (unsigned int i = 0; i < Entity::components.size(); i++){
//			std::map<int, std::vector<Component*>> *entityIdToComponentMap = &Entity::components[i];
//
//			std::map<int, std::vector<Component*>>::iterator it2 = entityIdToComponentMap->find(entity->entityId);
//			if (it2 != entityIdToComponentMap->end()){
//				std::vector<Component*> *cvec = &(it2->second);
//
//				for (unsigned int j = 0; j < cvec->size(); j++){
//					delete (*cvec)[j];
//				}
//
//				entityIdToComponentMap->erase(it2);
//			}
//		}
//
//		delete it->second;
//
//		entities.erase(entity->entityId);
//	}
//}
//
//Entity* Entity::Instantiate()
//{
//	Entity* entity = new Entity();
//
//	std::cout << "entity instantiate called for entity with id " << entity->entityId << std::endl;
//
//	entities[entity->entityId] = entity;
//	
//	return entity;
//}
//
//std::map<std::type_index, int> Entity::createIndiciesToComponentMaps()
//{
//	std::map < std::type_index, int > a;
//	a[std::type_index(typeid(Transform))] = ComponentHandle::TransformHandle;
//	a[std::type_index(typeid(Mesh))] = ComponentHandle::MeshHandle;
//	a[std::type_index(typeid(ParticleMesh))] = ComponentHandle::ParticleMeshHandle;
//	a[std::type_index(typeid(DirectionalLight))] = ComponentHandle::DirectionalLightHandle;
//	a[std::type_index(typeid(SpotLight))] = ComponentHandle::SpotLightHandle;
//	a[std::type_index(typeid(PointLight))] = ComponentHandle::PointLightHandle;
//	a[std::type_index(typeid(FluidParticles))] = ComponentHandle::FluidParticlesHandle;
//	a[std::type_index(typeid(Particles))] = ComponentHandle::ParticlesHandle;
//	a[std::type_index(typeid(ClothParticles))] = ComponentHandle::ClothParticlesHandle;
//	a[std::type_index(typeid(Skybox))] = ComponentHandle::SkyboxHandle;
//	a[std::type_index(typeid(Rigidbody))] = ComponentHandle::RigidbodyHandle;
//	a[std::type_index(typeid(FPSCamera))] = ComponentHandle::FPSCameraHandle;
//	a[std::type_index(typeid(EditorCamera))] = ComponentHandle::EditorCameraHandle;
//	a[std::type_index(typeid(BoxCollider))] = ComponentHandle::BoxColliderHandle;
//	a[std::type_index(typeid(SphereCollider))] = ComponentHandle::SphereColliderHandle;
//
//	return a;
//}
//
//std::vector<std::map<int, std::vector<Component*> >> Entity::createComponentMaps()
//{
//	std::map<int, std::vector<Component*>> a[15];
//	std::vector<std::map<int, std::vector<Component*> >> b;
//	b.push_back(a[ComponentHandle::TransformHandle]);
//	b.push_back(a[ComponentHandle::MeshHandle]);
//	b.push_back(a[ComponentHandle::ParticleMeshHandle]);
//	b.push_back(a[ComponentHandle::DirectionalLightHandle]);
//	b.push_back(a[ComponentHandle::SpotLightHandle]);
//	b.push_back(a[ComponentHandle::PointLightHandle]);
//	b.push_back(a[ComponentHandle::FluidParticlesHandle]);
//	b.push_back(a[ComponentHandle::ParticlesHandle]);
//	b.push_back(a[ComponentHandle::ClothParticlesHandle]);
//	b.push_back(a[ComponentHandle::SkyboxHandle]);
//	b.push_back(a[ComponentHandle::RigidbodyHandle]);
//	b.push_back(a[ComponentHandle::FPSCameraHandle]);
//	b.push_back(a[ComponentHandle::EditorCameraHandle]);
//	b.push_back(a[ComponentHandle::BoxColliderHandle]);
//	b.push_back(a[ComponentHandle::SphereColliderHandle]);
//
//	return b;
//}