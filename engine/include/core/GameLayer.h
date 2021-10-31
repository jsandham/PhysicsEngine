#ifndef GAME_LAYER_H__
#define GAME_LAYER_H__

#include <string>

#include "Time.h"
#include "World.h"
#include "Layer.h"

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
    void update(const Time& time) override;
    void end() override;
};
} // namespace PhysicsEngine

#endif