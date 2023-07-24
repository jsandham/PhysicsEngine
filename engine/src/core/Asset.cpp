// #include <fstream>
// #include <iostream>
// #include <string>

// #include "../../include/core/Asset.h"
// #include "../../include/core/World.h"

// using namespace PhysicsEngine;

// Asset::Asset(World *world, const Id &id) : Object(world, id)
// {
//     mName = "Unnamed Asset";
// }

// Asset::Asset(World *world, const Guid &guid, const Id &id) : Object(world, guid, id)
// {
//     mName = "Unnamed Asset";
// }

// Asset::~Asset()
// {
// }

// void Asset::serialize(YAML::Node &out) const
// {
//     Object::serialize(out);
//     out["name"] = mName;
// }

// void Asset::deserialize(const YAML::Node &in)
// {
//     Object::deserialize(in);
//     mName = YAML::getValue<std::string>(in, "name");
// }

// bool Asset::writeToYAML(const std::string &filepath) const
// {
//     std::ofstream out;
//     out.open(filepath);

//     if (!out.is_open())
//     {
//         return false;
//     }

//     if (mHide == HideFlag::None)
//     {
//         YAML::Node n;
//         serialize(n);

//         YAML::Node assetNode;
//         assetNode[getObjectName()] = n;

//         out << assetNode;
//         out << "\n";
//     }
//     out.close();

//     return true;
// }

// void Asset::loadFromYAML(const std::string &filepath)
// {
//     YAML::Node in = YAML::LoadFile(filepath);

//     if (!in.IsMap())
//     {
//         return;
//     }

//     for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
//     {
//         if (it->first.IsScalar() && it->second.IsMap())
//         {
//             deserialize(it->second);
//         }
//     }
// }

// std::string Asset::getName() const
// {
//     return mName;
// }

// void Asset::setName(const std::string &name)
// {
//     mName = name;
// }

// bool Asset::isInternal(int type)
// {
//     return type >= PhysicsEngine::MIN_INTERNAL_ASSET && type <= PhysicsEngine::MAX_INTERNAL_ASSET;
// }