//#ifndef WORLD_LOADER_H__
//#define WORLD_LOADER_H__
//
//#include <string>
//
// namespace PhysicaEngine
//{
//	class World;
//
//	class WorldLoader
//	{
//	public:
//		WorldLoader();
//		~WorldLoader();
//
//		bool loadPNG(World* world, const std::string& filepath);
//		bool loadOBJ(World* world, const std::string& filepath);
//		bool loadShader(World* world, const std::string& filepath);
//		bool loadMaterial(World* world, const std::string& filepath);
//
//		bool loadSceneFromBinary(World* world, const std::string& filepath) const;
//		bool loadSceneFromYAML(World* world, const std::string& filepath) const;
//		bool writeSceneToBinary(const World* world, const std::string& filepath) const;
//		bool writeSceneToYAML(const World* world, const std::string& filepath) const;
//
//
//		/*bool loadAssetFromBinary(World* world, const std::string& filepath) const;
//		bool writeAssetToBinary(const World* world, const std::string& filepath) const;
//		bool writeSceneToBinary(const World* world, const std::string& filepath) const;
//		bool writeSceneToYAML(const World* world, const std::string& filepath) const;*/
//
//		bool isAssetHeaderValid() const;
//		bool isSceneHeaderValid() const;
//	};
//}
//
//#endif