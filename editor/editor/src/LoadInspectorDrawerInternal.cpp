#include <iostream>

#include "../include/LoadInspectorDrawerInternal.h"

#include "../include/TransformDrawer.h"
#include "../include/CameraDrawer.h"
#include "../include/LightDrawer.h"
#include "../include/RigidbodyDrawer.h"
#include "../include/MeshRendererDrawer.h"
#include "../include/LineRendererDrawer.h"
#include "../include/BoxColliderDrawer.h"
#include "../include/SphereColliderDrawer.h"
#include "../include/CapsuleColliderDrawer.h"
#include "../include/MeshColliderDrawer.h"

#include "../include/ShaderDrawer.h"
#include "../include/Texture2DDrawer.h"
#include "../include/Texture3DDrawer.h"
#include "../include/CubemapDrawer.h"
#include "../include/MaterialDrawer.h"
#include "../include/MeshDrawer.h"
#include "../include/FontDrawer.h"

#include "core/Log.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

InspectorDrawer* PhysicsEditor::loadInternalInspectorComponentDrawer(int type)
{
	if (type == 0){
		return new TransformDrawer();
	}
	else if (type == 1){
		return new RigidbodyDrawer();
	}
	else if (type == 2){
		return new CameraDrawer();
	}
	else if (type == 3){
		return new MeshRendererDrawer();
	}
	else if (type == 4){
		return new LineRendererDrawer();
	}
	else if (type == 5){
		return new LightDrawer();
	}
	else if (type == 8){
		return new BoxColliderDrawer();
	}
	else if (type == 9){
		return new SphereColliderDrawer();
	}
	else if (type == 10) {
		return new CapsuleColliderDrawer();
	}
	else if (type == 15){
		return new MeshColliderDrawer();
	}
	else{
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal inspector component drawer\n";
		Log::error(message.c_str());
		return NULL;
	}
}

InspectorDrawer* PhysicsEditor::loadInternalInspectorAssetDrawer(int type)
{
	if (type == 0) {
		return new ShaderDrawer();
	}
	else if (type == 1) {
		return new Texture2DDrawer();
	}
	else if (type == 2) {
		return new Texture3DDrawer();
	}
	else if (type == 3) {
		return new CubemapDrawer();
	}
	else if (type == 4) {
		return new MaterialDrawer();
	}
	else if (type == 5) {
		return new MeshDrawer();
	}
	else if (type == 6) {
		return new FontDrawer();
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal inspector asset drawer\n";
		Log::error(message.c_str());
		return NULL;
	}
}