#include <iostream>

#include "../include/LoadInspectorDrawerInternal.h"

#include "../include/TransformDrawer.h"
#include "../include/CameraDrawer.h"
#include "../include/LightDrawer.h"

using namespace PhysicsEditor;

InspectorDrawer* PhysicsEditor::loadInternalInspectorDrawer(int type)
{
	if (type == 0){
		return new TransformDrawer();
	}
	else if (type == 1){
		return NULL;//new RigidbodyDrawer();
	}
	else if (type == 2){
		return new CameraDrawer();
	}
	else if (type == 3){
		return NULL;//new MeshRendererDrawer();
	}
	else if (type == 4){
		return NULL;//new LineRendererDrawer();
	}
	else if (type == 5){
		return new LightDrawer();
	}
	else if (type == 8){
		return NULL;//new BoxColliderDrawer();
	}
	else if (type == 9){
		return NULL;//new SphereColliderDrawer();
	}
	else if (type == 15){
		return NULL;//new MeshColliderDrawer();
	}
	else if (type == 10){
		return NULL;//new CapsuleColliderDrawer();
	}
	else{
		std::cout << "Error: Invalid component type (" << type << ") when trying to load internal component inspector drawer" << std::endl;
		return NULL;
	}
}