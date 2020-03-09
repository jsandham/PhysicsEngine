#include "../../include/graphics/BatchManager.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

BatchManager::BatchManager()
{
	this->maxNumOfVerticesPerBatch = 100000; // ~1.1 mb
	this->maxNumOfMeshesPerBatch = 200;
}

BatchManager::BatchManager(unsigned int maxNumOfVerticesPerBatch, unsigned int maxNumOfMeshesPerBatch)
{
	this->maxNumOfVerticesPerBatch = maxNumOfVerticesPerBatch;
	this->maxNumOfMeshesPerBatch = maxNumOfMeshesPerBatch;
}

BatchManager::~BatchManager()
{

}

void BatchManager::add(Material* material, Mesh* mesh, glm::mat4 model)
{
	Guid materialId = material->getId();

	unsigned int numOfVerticesInMesh = (unsigned int)mesh->getVertices().size() / 3;

	std::map<Guid, std::vector<Batch>>::iterator it = materialIdToBatchesMap.find(materialId);
	if(it != materialIdToBatchesMap.end()){
		std::vector<Batch>& batches = it->second;

		for(size_t i = 0; i < batches.size(); i++){
			if(batches[i].hasEnoughRoom(numOfVerticesInMesh)){
				batches[i].add(mesh, model);
				return;
			}
		}	

		Batch batch;
		batch.maxNumOfVertices = maxNumOfVerticesPerBatch;
		batch.maxNumOfMeshes = maxNumOfMeshesPerBatch;
		batch.materialId = materialId;
		if(batch.hasEnoughRoom(numOfVerticesInMesh)){
			batch.generate();
			batch.add(mesh, model);

			batches.push_back(batch);
		}
	}
	else{
		Batch batch;
		batch.maxNumOfVertices = maxNumOfVerticesPerBatch;
		batch.maxNumOfMeshes = maxNumOfMeshesPerBatch;
		batch.materialId = materialId;
		if(batch.hasEnoughRoom(numOfVerticesInMesh)){
			batch.generate();
			batch.add(mesh, model);

			std::vector<Batch> batches;
			batches.push_back(batch);

			materialIdToBatchesMap[materialId] = batches;
		}
	}
}

void BatchManager::render(World* world, int variant, GraphicsQuery* query)
{
	std::map<Guid, std::vector<Batch>>::iterator it;
	for(it = materialIdToBatchesMap.begin(); it != materialIdToBatchesMap.end(); it++){
		std::vector<Batch> batches = it->second;

		if(query != NULL){
			query->numBatchDrawCalls += (unsigned int)batches.size();
		}

		for(size_t i = 0; i < batches.size(); i++){
			Guid materialId = batches[i].materialId;

			Material* material = world->getAsset<Material>(materialId);

			Graphics::render(world, material, variant, glm::mat4( 1.0 ), batches[i].VAO, batches[i].currentNumOfVertices, query);
		}
	}
}

void BatchManager::render(World* world, Material* material, int variant, GraphicsQuery* query)
{
	std::map<Guid, std::vector<Batch>>::iterator it;
	for(it = materialIdToBatchesMap.begin(); it != materialIdToBatchesMap.end(); it++){
		std::vector<Batch> batches = it->second;

		if(query != NULL){
			query->numBatchDrawCalls += (unsigned int)batches.size();
		}

		for(size_t i = 0; i < batches.size(); i++){
			Graphics::render(world, material, variant, glm::mat4( 1.0 ), batches[i].VAO, batches[i].currentNumOfVertices, query);
		}
	}
}

void BatchManager::render(World* world, Shader* shader, int variant, GraphicsQuery* query)
{
	std::map<Guid, std::vector<Batch>>::iterator it;
	for(it = materialIdToBatchesMap.begin(); it != materialIdToBatchesMap.end(); it++){
		std::vector<Batch> batches = it->second;

		if(query != NULL){
			query->numBatchDrawCalls += (unsigned int)batches.size();
		}

		for(size_t i = 0; i < batches.size(); i++){
			Graphics::render(world, shader, variant, glm::mat4( 1.0 ), batches[i].VAO, GL_TRIANGLES, batches[i].currentNumOfVertices, query);
		}
	}
}