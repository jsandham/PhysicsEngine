#include "../../include/core/AssetDatabase.h"

#include "../../include/json/json.hpp" 

using namespace PhysicsEngine;

AssetDatabase::AssetDatabase()
{

}

AssetDatabase::~AssetDatabase()
{

}

void AssetDatabase::add(std::string assetPath)
{
	// std::string jsonAssetFilePath = assetFiles[i].filepath.substr(0, assetFiles[i].filepath.find_last_of(".")) + ".json";
	// std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
	// std::ostringstream contents; contents << in.rdbuf(); in.close();

	// json::JSON jsonAsset = JSON::Load(contents.str());

	// Guid assetId = jsonAsset["id"].ToString();

	// std::map<Guid, std::string>::iterator it = assetIdToFilePath.find(assetId);
	// if(it == assetIdToFilePath.end()){
	// 	assetIdToFilePath[assetId] = assetFiles[i].filepath;
	// 	std::cout << "asset file: " << assetFiles[i].filepath << std::endl;
	// }



	// for(unsigned int i = 0; i < assetFiles.size(); i++){
	// 	std::string jsonAssetFilePath = assetFiles[i].filepath.substr(0, assetFiles[i].filepath.find_last_of(".")) + ".json";
	// 	std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
	// 	std::ostringstream contents; contents << in.rdbuf(); in.close();

	// 	json::JSON jsonAsset = JSON::Load(contents.str());

	// 	Guid assetId = jsonAsset["id"].ToString();

	// 	std::map<Guid, std::string>::iterator it = assetIdToFilePath.find(assetId);
	// 	if(it == assetIdToFilePath.end()){
	// 		assetIdToFilePath[assetId] = assetFiles[i].filepath;
	// 		std::cout << "asset file: " << assetFiles[i].filepath << std::endl;
	// 	}
	// }
}

std::string AssetDatabase::get(Guid guid)
{
	return "";
}