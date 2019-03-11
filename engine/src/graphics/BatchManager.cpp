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

void BatchManager::add(Material* material, Mesh* mesh)
{
	Guid materialId = material->assetId;

	unsigned int numOfVerticesInMesh = (unsigned int)mesh->vertices.size() / 3;

	std::map<Guid, std::vector<Batch>>::iterator it = materialIdToBatchesMap.find(materialId);
	if(it != materialIdToBatchesMap.end()){
		std::vector<Batch>& batches = it->second;

		for(size_t i = 0; i < batches.size(); i++){
			if(batches[i].hasEnoughRoom(numOfVerticesInMesh)){
				batches[i].add(mesh);
				return;
			}
		}	

		Batch batch;
		batch.maxNumOfVertices = maxNumOfVerticesPerBatch;
		batch.maxNumOfMeshes = maxNumOfMeshesPerBatch;
		batch.materialId = materialId;
		if(batch.hasEnoughRoom(numOfVerticesInMesh)){
			batch.generate();
			batch.add(mesh);

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
			batch.add(mesh);

			std::vector<Batch> batches;
			batches.push_back(batch);

			materialIdToBatchesMap[materialId] = batches;
		}
	}
}

void BatchManager::render(World* world)
{
	std::map<Guid, std::vector<Batch>>::iterator it;
	for(it = materialIdToBatchesMap.begin(); it != materialIdToBatchesMap.end(); it++){
		std::vector<Batch> batches = it->second;

		for(size_t i = 0; i < batches.size(); i++){
			Guid materialId = batches[i].materialId;

			Material* material = world->getAsset<Material>(materialId);
			Shader* shader = world->getAsset<Shader>(material->shaderId);

			if(material == NULL){
				std::cout << "Material is NULL" << std::endl;
				return;
			}

			if(shader == NULL){
				std::cout << "Shader is NULL" << std::endl;
				return;
			}

			if(!shader->isCompiled()){
				std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
				return;
			}

			Graphics::use(shader);
			Graphics::setFloat(shader, "material.shininess", material->shininess);
			Graphics::setVec3(shader, "material.ambient", material->ambient);
			Graphics::setVec3(shader, "material.diffuse", material->diffuse);
			Graphics::setVec3(shader, "material.specular", material->specular);

			Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
			if(mainTexture != NULL){
				Graphics::setInt(shader, "material.mainTexture", 0);

				Graphics::active(mainTexture, 0);
				Graphics::bind(mainTexture);
			}

			Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
			if(normalMap != NULL){

				Graphics::setInt(shader, "material.normalMap", 1);

				Graphics::active(normalMap, 1);
				Graphics::bind(normalMap);
			}

			Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);
			if(specularMap != NULL){

				Graphics::setInt(shader, "material.specularMap", 2);

				Graphics::active(specularMap, 2);
				Graphics::bind(specularMap);
			}

			glBindVertexArray(batches[i].VAO);

			//std::cout << batches[i].currentNumOfVertices << "  " << batches[i].VAO << " " << batches[i].vertexVBO << " " << batches[i].normalVBO << " " << batches[i].texCoordVBO << " " << std::endl;

			glDrawArrays(GL_TRIANGLES, 0, batches[i].currentNumOfVertices);

			GLenum error;
			while ((error = glGetError()) != GL_NO_ERROR){
				std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
			}
		}
	}
}

void BatchManager::render(World* world, Material* material)
{

}