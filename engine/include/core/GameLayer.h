#ifndef GAME_LAYER_H__
#define GAME_LAYER_H__

#include <string>

#include "Layer.h"
#include "World.h"

namespace PhysicsEngine
{
class GameLayer : public Layer
{
  private:
    World mWorld;

  public:
    GameLayer();
    ~GameLayer();

    void init() override;
    void begin() override;
    void update() override;
    void end() override;
    bool quit() override;
};
} // namespace PhysicsEngine

#endif