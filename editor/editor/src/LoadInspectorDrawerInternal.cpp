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
	if (type == ComponentType<Transform>::type){
		return new TransformDrawer();
	}
	else if (type == ComponentType<Rigidbody>::type){
		return new RigidbodyDrawer();
	}
	else if (type == ComponentType<Camera>::type){
		return new CameraDrawer();
	}
	else if (type == ComponentType<MeshRenderer>::type){
		return new MeshRendererDrawer();
	}
	else if (type == ComponentType<LineRenderer>::type){
		return new LineRendererDrawer();
	}
	else if (type == ComponentType<Light>::type){
		return new LightDrawer();
	}
	else if (type == ComponentType<BoxCollider>::type){
		return new BoxColliderDrawer();
	}
	else if (type == ComponentType<SphereCollider>::type){
		return new SphereColliderDrawer();
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		return new CapsuleColliderDrawer();
	}
	else if (type == ComponentType<MeshCollider>::type){
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
	if (type == AssetType<Shader>::type) {
		return new ShaderDrawer();
	}
	else if (type == AssetType<Texture2D>::type) {
		return new Texture2DDrawer();
	}
	else if (type == AssetType<Texture3D>::type) {
		return new Texture3DDrawer();
	}
	else if (type == AssetType<Cubemap>::type) {
		return new CubemapDrawer();
	}
	else if (type == AssetType<Material>::type) {
		return new MaterialDrawer();
	}
	else if (type == AssetType<Mesh>::type) {
		return new MeshDrawer();
	}
	else if (type == AssetType<Font>::type) {
		return new FontDrawer();
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal inspector asset drawer\n";
		Log::error(message.c_str());
		return NULL;
	}
}