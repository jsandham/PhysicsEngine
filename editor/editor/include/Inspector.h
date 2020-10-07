#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "EditorClipboard.h"
#include "EditorProject.h"
#include "EditorScene.h"
#include "InspectorDrawer.h"

#include "MaterialDrawer.h"
#include "ShaderDrawer.h"
#include "Texture2DDrawer.h"

#include "core/Entity.h"
#include "core/World.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class Inspector
{
  private:
    MaterialDrawer materialDrawer;
    ShaderDrawer shaderDrawer;
    Texture2DDrawer texture2DDrawer;

  public:
    Inspector();
    ~Inspector();
    Inspector(const Inspector &other) = delete;
    Inspector &operator=(const Inspector &other) = delete;

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                bool isOpenedThisFrame);

  private:
    void drawEntity(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard);

    // template<class T>
    // void drawAsset(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard)
    //{
    //	if (AssetType<T>::type == AssetType<Material>::type) {
    //		materialDrawer->render(world, project, scene, clipboard, clipboard.getSelectedId());
    //	}
    //	else if (AssetType<T>::type == AssetType<Shader>::type) {
    //		shaderDrawer->render(world, project, scene, clipboard, clipboard.getSelectedId());
    //	}
    //	else if (AssetType<T>::type == AssetType<Texture2D>::type) {
    //		texture2DDrawer->render(world, project, scene, clipboard, clipboard.getSelectedId());
    //	}

    //	ImGui::Separator();

    //	/*T* asset = world->getAsset<T>(clipboard.getSelectedId());

    //	InspectorDrawer* drawer = loadInternalInspectorAssetDrawer(AssetType<T>::type);

    //	drawer->render(world, project, scene, clipboard, asset->assetId);
    //	ImGui::Separator();*/

    //	delete drawer;
    //}

    void drawCodeFile(World *world, EditorScene &scene, EditorClipboard &clipboard);

    // move to imgui extensions?
    bool BeginAddComponentDropdown(std::string filter, std::string &componentToAdd);
    void EndAddComponentDropdown();
};
} // namespace PhysicsEditor

#endif