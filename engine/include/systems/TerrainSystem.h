#ifndef TERRAINSYSTEM_H__
#define TERRAINSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

#include "../components/Camera.h"

namespace PhysicsEngine
{
    class TerrainSystem : public System
    {
    private:

    public:
        TerrainSystem(World *world);
        TerrainSystem(World *world, const Guid& id);
        ~TerrainSystem();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void init(World* world) override;
        void update(const Input& input, const Time& time) override;
    };

    template <> struct SystemType<TerrainSystem>
    {
        static constexpr int type = PhysicsEngine::TERRAINSYSTEM_TYPE;
    };
    template <> struct IsSystemInternal<TerrainSystem>
    {
        static constexpr bool value = true;
    };
} // namespace PhysicsEngine

#endif