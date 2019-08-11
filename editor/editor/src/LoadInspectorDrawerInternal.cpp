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

using namespace PhysicsEditor;

InspectorDrawer* PhysicsEditor::loadInternalInspectorDrawer(int type)
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
		std::cout << "Error: Invalid component type (" << type << ") when trying to load internal component inspector drawer" << std::endl;
		return NULL;
	}
}