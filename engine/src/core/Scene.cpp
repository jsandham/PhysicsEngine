#include "../../include/core/Scene.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Types.h"
#include "../../include/core/Version.h"

using namespace PhysicsEngine;

Scene::Scene(World *world) : Object(world)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::Scene(World *world, Guid id) : Object(world, id)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::~Scene()
{
}

void Scene::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["name"] = mName;
    out["version"] = mVersion;
}

void Scene::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

    mName = YAML::getValue<std::string>(in, "name");
    mVersion = YAML::getValue<std::string>(in, "version");
}

int Scene::getType() const
{
    return PhysicsEngine::SCENE_TYPE;
}

std::string Scene::getObjectName() const
{
    return PhysicsEngine::SCENE_NAME;
}

void Scene::load(const std::string &filepath)
{
    if (filepath.empty())
    {
        return;
    }

    /*YAML::Node in = YAML::LoadFile(filepath);

    if (!in.IsMap()) {
        return false;
    }

    mId = YAML::getValue<Guid>(in, "id");

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it) {
        if (it->first.IsScalar() && it->second.IsMap()) {
            if (loadSceneObjectFromYAML(it->second) == nullptr) {
                return false;
            }
        }
    }

    return true;*/
}

std::string Scene::getName() const
{
    return mName;
}