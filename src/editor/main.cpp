#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../entities/Entity.h"
#include "../components/Transform.h"
#include "../components/MeshRenderer.h"
#include "../components/DirectionalLight.h"

#include "../json/json.hpp"
using namespace json;
using namespace PhysicsEngine;

#pragma pack(push, 1)
	struct SceneHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
		unsigned int numberOfEntities;
		unsigned int numberOfTransforms;
		unsigned int numberOfMeshRenderers;
		unsigned int sizeOfEntity;
		unsigned int sizeOfTransform;
		unsigned int sizeOfMeshRenderer;
	};
#pragma pack(pop)

int main(int argc, char* argv[])
{
	if(argc > 0){
		std::cout << argv[0] << std::endl;
	}

	std::string filePath = "../data/scenes/simple.json";

	// open json file and load to json object
	std::ifstream in(filePath, std::ios::in | std::ios::binary);
	std::ostringstream contents;
	contents << in.rdbuf();
	in.close();
	std::string jsonString = contents.str();
	json::JSON scene = JSON::Load(jsonString);

	// create scene header
	SceneHeader header = {};

	header.sizeOfEntity = sizeof(Entity);
	header.sizeOfTransform = sizeof(Transform);
	header.sizeOfMeshRenderer = sizeof(MeshRenderer);







	std::cout << "size of entity pointer: " << sizeof(Entity*) << std::endl;
	std::cout << "size of bool: " << sizeof(bool) << std::endl;
	std::cout << "size of int: " << sizeof(int) << std::endl;
	std::cout << "size of transform: " << header.sizeOfTransform << std::endl;
	std::cout << "size of mesh renderer: " << header.sizeOfMeshRenderer << std::endl;
	std::cout << "size fo directional light: " << sizeof(DirectionalLight) << std::endl;

	std::cout << "size: " << scene.size() << std::endl;
	std::cout << scene["23424355"]["type"] << std::endl;

	// std::cout << "size of component array: " << scene["23424355"]["components"].size() << std::endl;
	// std::cout << scene["23424355"]["components"][0] << " " << scene["23424355"]["components"][1] << std::endl;
	json::JSON components = scene["23424355"]["components"];
	std::cout << "size of component array: " << components.size() << std::endl;
	for(int i = 0; i < components.size(); i++)
	{
		std::cout << components[i] << std::endl;
	}


	// std::cout << obj["obj"]["inner"] << std::endl;
	// std::cout << obj["new"]["some"]["deep"]["key"] << std::endl;
	// std::cout << obj.size() << std::endl;
	// std::cout << obj["array"][1] << std::endl;
	// std::cout << obj.hasKey("array") << std::endl;
	// std::cout << obj.hasKey("obj") << std::endl;
	// std::cout << obj.hasKey("inner") << std::endl;

	// if(obj["array"].JSONType()== JSON::Class::Array)
	// {
	// 	std::cout << "finding type works" << std::endl;
	// }



	// json::JSON obj;
	// // Create a new Array as a field of an Object.
	// obj["array"] = json::Array( true, "Two", 3, 4.0 );
	// // Create a new Object as a field of another Object.
	// obj["obj"] = json::Object();
	// // Assign to one of the inner object's fields
	// obj["obj"]["inner"] = "Inside";
	  
	// // We don't need to specify the type of the JSON object:
	// obj["new"]["some"]["deep"]["key"] = "Value";
	// obj["array2"].append( false, "three" );
	  
	// // We can also parse a string into a JSON object:
	// obj["parsed"] = JSON::Load( "[ { \"Key\" : \"Value\" }, false ]" );
  
  	//std::cout << obj << std::endl;


	while(true)
	{

	}

	return 0;
}