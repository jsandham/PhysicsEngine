#include <iostream>

#include "../../include/drawers/LoadInspectorDrawerInternal.h"

#include "../../include/drawers/BoxColliderDrawer.h"
#include "../../include/drawers/CameraDrawer.h"
#include "../../include/drawers/CapsuleColliderDrawer.h"
#include "../../include/drawers/LightDrawer.h"
#include "../../include/drawers/LineRendererDrawer.h"
#include "../../include/drawers/MeshColliderDrawer.h"
#include "../../include/drawers/MeshRendererDrawer.h"
#include "../../include/drawers/RigidbodyDrawer.h"
#include "../../include/drawers/SphereColliderDrawer.h"
#include "../../include/drawers/TransformDrawer.h"

#include "../../include/drawers/CubemapDrawer.h"
#include "../../include/drawers/FontDrawer.h"
#include "../../include/drawers/MaterialDrawer.h"
#include "../../include/drawers/MeshDrawer.h"
#include "../../include/drawers/ShaderDrawer.h"
#include "../../include/drawers/Texture2DDrawer.h"
#include "../../include/drawers/Texture3DDrawer.h"

#include "core/Log.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

InspectorDrawer *PhysicsEditor::loadInternalInspectorComponentDrawer(int type)
{
    if (type == ComponentType<Transform>::type)
    {
        return new TransformDrawer();
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        return new RigidbodyDrawer();
    }
    else if (type == ComponentType<Camera>::type)
    {
        return new CameraDrawer();
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        return new MeshRendererDrawer();
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        return new LineRendererDrawer();
    }
    else if (type == ComponentType<Light>::type)
    {
        return new LightDrawer();
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        return new BoxColliderDrawer();
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        return new SphereColliderDrawer();
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        return new CapsuleColliderDrawer();
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        return new MeshColliderDrawer();
    }
    else
    {
        std::string message = "Invalid component type (" + std::to_string(type) +
                              ") when trying to load internal inspector component drawer\n";
        Log::error(message.c_str());
        return NULL;
    }
}

InspectorDrawer *PhysicsEditor::loadInternalInspectorAssetDrawer(int type)
{
    if (type == AssetType<Shader>::type)
    {
        return new ShaderDrawer();
    }
    else if (type == AssetType<Texture2D>::type)
    {
        return new Texture2DDrawer();
    }
    else if (type == AssetType<Texture3D>::type)
    {
        return new Texture3DDrawer();
    }
    else if (type == AssetType<Cubemap>::type)
    {
        return new CubemapDrawer();
    }
    else if (type == AssetType<Material>::type)
    {
        return new MaterialDrawer();
    }
    else if (type == AssetType<Mesh>::type)
    {
        return new MeshDrawer();
    }
    else if (type == AssetType<Font>::type)
    {
        return new FontDrawer();
    }
    else
    {
        std::string message =
            "Invalid asset type (" + std::to_string(type) + ") when trying to load internal inspector asset drawer\n";
        Log::error(message.c_str());
        return NULL;
    }
}