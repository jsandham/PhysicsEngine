#include <iostream>

#include "../../include/core/SlabBuffer.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

int SlabBuffer::test = 0;

SlabBuffer::SlabBuffer()
{
	root = NULL;
	next = NULL;
}

SlabBuffer::~SlabBuffer()
{
	SlabNode* current = root;
	SlabNode* previous = NULL;
	while(current != NULL){
		previous = current;
		current = current->next;

		delete previous;
	}
}

void SlabBuffer::clear()
{
	next = root;

	SlabNode* current = root;
	while(current != NULL){
		current->numberOfLinesToDraw = 0;
		current = current->next;
	}
}

void SlabBuffer::add(glm::vec3 start, glm::vec3 end, Material* material)
{
	SlabNode** current = &root;

	bool hasAdded = false;
	while(!hasAdded){

		if(*current != NULL){
			if(material->assetId == (*current)->material->assetId){

				(*current)->material = material;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw] = start.x;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw + 1] = start.y;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw + 2] = start.z;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw + 3] = end.x;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw + 4] = end.y;
				(*current)->buffer[6*(*current)->numberOfLinesToDraw + 5] = end.z;
				(*current)->numberOfLinesToDraw++;

				hasAdded = true;
			}
			else{
				*current = (*current)->next;
			}
		}
		else{
			SlabBuffer::test++;
			std::cout << "creating new slab node " << SlabBuffer::test << std::endl;
			*current = new SlabNode();
			(*current)->material = material;
			(*current)->buffer[0] = start.x;
			(*current)->buffer[1] = start.y;
			(*current)->buffer[2] = start.z;
			(*current)->buffer[3] = end.x;
			(*current)->buffer[4] = end.y;
			(*current)->buffer[5] = end.z;
			(*current)->numberOfLinesToDraw = 1;

			Graphics::generate(*current);

			hasAdded = true;
		}
	}
}

void SlabBuffer::add(std::vector<float> lines, Material* material)
{
	SlabNode** current = &root;

	bool hasAdded = false;
	while(!hasAdded){

		if(*current != NULL){
			if(material->assetId == (*current)->material->assetId){

				(*current)->material = material;
				for(unsigned int i = 0; i < lines.size(); i++){
					(*current)->buffer[6*(*current)->numberOfLinesToDraw + i] = lines[i];
				}
				
				(*current)->numberOfLinesToDraw += (int)lines.size() / 6;

				hasAdded = true;
			}
			else{
				*current = (*current)->next;
			}
		}
		else{
			SlabBuffer::test++;
			std::cout << "creating new slab node " << SlabBuffer::test << std::endl;
			*current = new SlabNode();
			(*current)->material = material;
			for(unsigned int i = 0; i < lines.size(); i++){
				(*current)->buffer[i] = lines[i];
			}
			(*current)->numberOfLinesToDraw = (int)lines.size() / 6;

			Graphics::generate(*current);

			hasAdded = true;
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