//#include "WorldLoader"
//
// using namespace PhysicsEngine;
//
// WorldLoader::WorldLoader()
//{
//
//}
//
// WorldLoader::~WorldLoader()
//{
//
//}
//
// bool WorldLoader::loadAssetFromBinary(WorldAllocators* world, const std::string& filepath) const
//{
//    std::ifstream file;
//    file.open(filePath, std::ios::binary);
//
//    if (!file.is_open())
//    {
//        std::string errorMessage = "Failed to open asset bundle " + filePath + "\n";
//        Log::error(&errorMessage[0]);
//        return false;
//    }
//
//    AssetHeader assetHeader;
//    PhysicsEngine::read<AssetHeader>(file, assetHeader);
//
//    assert(assetHeader.mSignature == ASSET_FILE_SIGNATURE && "Trying to load an invalid binary asset file\n");
//
//    ObjectHeader header;
//    PhysicsEngine::read<ObjectHeader>(file, header);
//
//    loadAssetFromBinary(file, header);
//
//    file.close();
//
//    return true;
//}
//
// bool WorldLoader::loadSceneFromBinary(WorldAllocators* world, const std::string& filepath) const
//{
//    std::ifstream file;
//    file.open(filePath, std::ios::binary);
//
//    if (!file.is_open())
//    {
//        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
//        Log::error(&errorMessage[0]);
//        return false;
//    }
//
//    SceneHeader sceneHeader;
//    PhysicsEngine::read<SceneHeader>(file, sceneHeader);
//
//    assert(sceneHeader.mSignature == SCENE_FILE_SIGNATURE && "Trying to load an invalid binary scene file\n");
//
//    while (file.peek() != EOF)
//    {
//        std::cout << "loading i: " << std::endl;
//
//        ObjectHeader header;
//        PhysicsEngine::read<ObjectHeader>(file, header);
//
//        std::cout << "loading id: " << header.mId.toString() << std::endl;
//        std::cout << "loading type: " << header.mType << std::endl;
//        std::cout << "loading is internal: " << header.mIsTnternal << std::endl;
//
//        if (!loadBinary(file, header))
//        {
//            std::cout << "Error occured" << std::endl;
//            break;
//        }
//    }
//
//    file.close();
//
//    return true;
//}
//
// bool WorldLoader::loadSceneFromYAML(WorldAllocators* world, const std::string& filepath) const
//{
//    YAML::Node in = YAML::LoadFile(filePath);
//
//    if (!in.IsMap()) {
//        return false;
//    }
//
//    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it) {
//        if (it->first.IsScalar()) {
//
//            if (!loadYAML(it->second)) {
//                return false;
//            }
//        }
//    }
//
//    return false;
//}
//
// bool WorldLoader::writeAssetToBinary(const WorldAllocators* world, const std::string& filepath) const
//{
//
//}
//
// bool WorldLoader::writeSceneToBinary(const WorldAllocators* world, const std::string& filepath) const
//{
//
//}
//
// bool WorldLoader::writeSceneToYAML(const WorldAllocators* world, const std::string& filepath) const
//{
//    std::ofstream out;
//    out.open(filePath);
//
//    if (!out.is_open()) {
//        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
//        Log::error(&errorMessage[0]);
//        return false;
//    }
//
//    for (size_t i = 0; i < getNumberOfEntities(); i++) {
//        Entity* entity = getEntityByIndex(i);
//
//        YAML::Node en;
//        entity->serialize(en);
//
//        YAML::Node entityNode;
//        entityNode[entity->getObjectName()] = en;
//
//        out << entityNode;
//        out << "\n";
//
//        std::vector<std::pair<Guid, int>> temp = entity->getComponentsOnEntity(this);
//        for (size_t j = 0; j < temp.size(); j++) {
//            Component* component = nullptr;
//
//            if (Component::isInternal(temp[j].second))
//            {
//                component = PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
//            }
//            else
//            {
//                component = PhysicsEngine::getComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
//            }
//
//            YAML::Node cn;
//            component->serialize(cn);
//
//            YAML::Node componentNode;
//            componentNode[component->getObjectName()] = cn;
//
//            out << componentNode;
//            out << "\n";
//        }
//    }
//
//    out.close();
//
//    return true;
//}
//
// bool WorldLoader::isAssetHeaderValid() const
//{
//
//}
//
// bool WorldLoader::isSceneHeaderValid() const
//{
//
//}