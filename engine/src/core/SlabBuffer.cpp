#include <iostream>

#include "../../include/core/SlabBuffer.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

int SlabBuffer::test = 0;

SlabBuffer::SlabBuffer(size_t blockSize)
{
	root = NULL;
	next = NULL;

	this->blockSize = blockSize;
}

SlabBuffer::~SlabBuffer()
{
	SlabNode* current = root;
	SlabNode* previous = NULL;
	while(current != NULL){
		previous = current;
		current = current->next;

		delete [] previous->buffer;
		delete previous;
	}
}

void SlabBuffer::clear()
{
	next = root;

	SlabNode* current = root;
	while(current != NULL){
		current->count = 0;
		current = current->next;
	}
}

// void SlabBuffer::add(glm::vec3 start, glm::vec3 end, Material* material)
// {
// 	SlabNode** current = &root;

// 	bool hasAdded = false;
// 	while(!hasAdded){

// 		if(*current != NULL){
// 			if(material->assetId == (*current)->material->assetId && 6*((*current)->count + 1) < blockSize){

// 				(*current)->material = material;
// 				(*current)->buffer[6*(*current)->count] = start.x;
// 				(*current)->buffer[6*(*current)->count + 1] = start.y;
// 				(*current)->buffer[6*(*current)->count + 2] = start.z;
// 				(*current)->buffer[6*(*current)->count + 3] = end.x;
// 				(*current)->buffer[6*(*current)->count + 4] = end.y;
// 				(*current)->buffer[6*(*current)->count + 5] = end.z;
// 				(*current)->numberOfLinesToDraw++;

// 				hasAdded = true;
// 			}
// 			else{
// 				*current = (*current)->next;
// 			}
// 		}
// 		else{
// 			SlabBuffer::test++;
// 			std::cout << "creating new slab node " << SlabBuffer::test << std::endl;
// 			*current = new SlabNode();
// 			(*current)->material = material;
// 			(*current)->buffer = new float[blockSize];

// 			(*current)->buffer[0] = start.x;
// 			(*current)->buffer[1] = start.y;
// 			(*current)->buffer[2] = start.z;
// 			(*current)->buffer[3] = end.x;
// 			(*current)->buffer[4] = end.y;
// 			(*current)->buffer[5] = end.z;

// 			(*current)->count = 1;

// 			Graphics::generate(*current);

// 			hasAdded = true;
// 		}
// 	}
// }


void SlabBuffer::add(std::vector<float> data, Material* material)
{
	size_t startIndex = 0;
	size_t endIndex = data.size();

	SlabNode** current = &root;

	while(startIndex < endIndex){

		if(*current != NULL){
			size_t count = (*current)->count;
			if(material->assetId == (*current)->material->assetId && count < blockSize){

				if((count + endIndex - startIndex) > blockSize){
					endIndex = startIndex + blockSize - count;
				}

				(*current)->material = material;
				for(size_t i = 0; i < endIndex-startIndex; i++){
					(*current)->buffer[count + i] = data[startIndex + i];
				}
				
				(*current)->count += (endIndex - startIndex);

				startIndex = endIndex;
				endIndex = data.size();

				//std::cout << "start index: " << startIndex << " end index: " << endIndex << std::endl;
			}
			else{
				current = &((*current)->next);
			}
		}
		else{
			SlabBuffer::test++;
			std::cout << "creating new slab node " << SlabBuffer::test << std::endl;
			*current = new SlabNode();
			(*current)->material = material;
			(*current)->buffer = new float[blockSize];
			(*current)->size = blockSize;


			if((endIndex - startIndex) > blockSize){
				endIndex = startIndex + blockSize;
			}

			for(size_t i = 0; i < endIndex-startIndex; i++){
				(*current)->buffer[i] = data[startIndex + i];
			}

			(*current)->count = (endIndex - startIndex);

			Graphics::generate(*current);

			startIndex = endIndex;
			endIndex = data.size();
		}
	}
}

bool SlabBuffer::hasNext()
{
	return next != NULL;
}

SlabNode* SlabBuffer::getNext()
{
	SlabNode* temp = next;
	next = next->next;

	return temp;
}

size_t SlabBuffer::getBlockSize()
{
	return blockSize;
}