#ifndef SYSTEM_TYPES_H__
#define SYSTEM_TYPES_H__

#include "../core/Types.h"

#include "AssetLoadingSystem.h"
#include "CleanUpSystem.h"
#include "DebugSystem.h"
#include "FreeLookCameraSystem.h"
#include "GizmoSystem.h"
#include "PhysicsSystem.h"
#include "RenderSystem.h"
#include "TerrainSystem.h"

namespace PhysicsEngine
{
    template <typename T> struct SystemType
    {
        static constexpr int type = PhysicsEngine::INVALID_TYPE;
    };

    template <> struct SystemType<AssetLoadingSystem>
    {
        static constexpr int type = PhysicsEngine::ASSETLOADINGSYSTEM_TYPE;
    };

    template <> struct SystemType<CleanUpSystem>
    {
        static constexpr int type = PhysicsEngine::CLEANUPSYSTEM_TYPE;
    };

    template <> struct SystemType<DebugSystem>
    {
        static constexpr int type = PhysicsEngine::DEBUGSYSTEM_TYPE;
    };

    template <> struct SystemType<FreeLookCameraSystem>
    {
        static constexpr int type = PhysicsEngine::FREELOOKCAMERASYSTEM_TYPE;
    };

    template <> struct SystemType<GizmoSystem>
    {
        static constexpr int type = PhysicsEngine::GIZMOSYSTEM_TYPE;
    };

    template <> struct SystemType<PhysicsSystem>
    {
        static constexpr int type = PhysicsEngine::PHYSICSSYSTEM_TYPE;
    };

    template <> struct SystemType<RenderSystem>
    {
        static constexpr int type = PhysicsEngine::RENDERSYSTEM_TYPE;
    };

    template <> struct SystemType<TerrainSystem>
    {
        static constexpr int type = PhysicsEngine::TERRAINSYSTEM_TYPE;
    };
}

#endif