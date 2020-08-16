#include <iostream>
#include <vector>

#include <core/WriteToJson.h>

using namespace PhysicsEngine;

void PhysicsEngine::writeComponentToJson(json::JSON& obj, World* world, Guid entityId, Guid componentId, int type)
{
	if (Component::isInternal(type)) {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to write internal component to json\n";
		Log::error(message.c_str());
		return;
	}

	return;
}

void PhysicsEngine::writeSystemToJson(json::JSON& obj, World* world, Guid systemId, int type, int order)
{
	if (System::isInternal(type)) {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to write internal system to json\n";
		Log::error(message.c_str());
		return;
	}

	//if (type == 21) {
	//	// EditorCameraSystem
	//	obj[systemId.toString()]["type"] = "EditorCameraSystem";
	//	obj[systemId.toString()]["order"] = order;

	//}
	//else {
	//	std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
	//	Log::error(message.c_str());
	//	return;
	//}

	return;
}