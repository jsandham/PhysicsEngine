#ifndef WORLDMANAGER_H__
#define WORLDMANAGER_H__

#include <string>

#include "Input.h"
#include "Time.h"
#include "World.h"

namespace PhysicsEngine
{
class WorldManager
{
  private:
    World world;

  public:
    WorldManager();
    ~WorldManager();

    bool load(std::string sceneFilePath, std::vector<std::string> assetFilePaths);
    void init();
    void update(Time time, Input input);
};
} // namespace PhysicsEngine

#endif