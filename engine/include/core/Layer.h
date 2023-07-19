#ifndef LAYER_H__
#define LAYER_H__

#include <string>

namespace PhysicsEngine
{
class Layer
{
  private:
    std::string mName;

  public:
    Layer(const std::string &name = "Layer");
    virtual ~Layer() = 0;

    virtual void init() = 0;
    virtual void begin() = 0;
    virtual void update() = 0;
    virtual void end() = 0;
    virtual bool quit() = 0;
};
} // namespace PhysicsEngine

#endif