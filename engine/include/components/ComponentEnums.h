#ifndef COMPONENT_ENUMS_H__
#define COMPONENT_ENUMS_H__

namespace PhysicsEngine
{
    // Light
    enum class CameraMode
    {
        Main,
        Secondary
    };

    enum class CameraSSAO
    {
        SSAO_On,
        SSAO_Off,
    };

    enum class CameraGizmos
    {
        Gizmos_On,
        Gizmos_Off,
    };

    enum class RenderPath
    {
        Forward,
        Deferred
    };

    enum class ColorTarget
    {
        Color,
        Normal,
        Position,
        LinearDepth,
        ShadowCascades
    };

    enum class ShadowCascades
    {
        NoCascades = 0,
        TwoCascades = 1,
        ThreeCascades = 2,
        FourCascades = 3,
        FiveCascades = 4,
    };

    constexpr auto CameraModeToString(CameraMode mode)
    {
        switch (mode)
        {
        case CameraMode::Main:
            return "Main";
        case CameraMode::Secondary:
            return "Secondary";
        }
    }

    constexpr auto CameraSSAOToString(CameraSSAO ssao)
    {
        switch (ssao)
        {
        case CameraSSAO::SSAO_On:
            return "SSAO On";
        case CameraSSAO::SSAO_Off:
            return "SSAO Off";
        }
    }

    constexpr auto CameraGizmosToString(CameraGizmos gizmo)
    {
        switch (gizmo)
        {
        case CameraGizmos::Gizmos_On:
            return "Gizmos On";
        case CameraGizmos::Gizmos_Off:
            return "Gizmos Off";
        }
    }

    constexpr auto RenderPathToString(RenderPath renderPath)
    {
        switch (renderPath)
        {
        case RenderPath::Forward:
            return "Forward";
        case RenderPath::Deferred:
            return "Deferred";
        }
    }

    constexpr auto ColorTargetToString(ColorTarget target)
    {
        switch (target)
        {
        case ColorTarget::Color:
            return "Color";
        case ColorTarget::LinearDepth:
            return "LinearDepth";
        case ColorTarget::Normal:
            return "Normal";
        case ColorTarget::Position:
            return "Position";
        case ColorTarget::ShadowCascades:
            return "ShadowCascades";
        }
    }

    constexpr auto ShadowCascadesToString(ShadowCascades cascade)
    {
        switch (cascade)
        {
        case ShadowCascades::NoCascades:
            return "NoCascades";
        case ShadowCascades::TwoCascades:
            return "TwoCascades";
        case ShadowCascades::ThreeCascades:
            return "ThreeCascades";
        case ShadowCascades::FourCascades:
            return "FourCascades";
        case ShadowCascades::FiveCascades:
            return "FiveCascades";
        }
    }

    // Shader
    enum class LightType
    {
        Directional,
        Spot,
        Point,
        None
    };

    enum class ShadowType
    {
        Hard,
        Soft,
        None
    };

    enum class ShadowMapResolution
    {
        Low512x512 = 512,
        Medium1024x1024 = 1024,
        High2048x2048 = 2048,
        VeryHigh4096x4096 = 4096
    };

    constexpr auto LightTypeToString(LightType type)
    {
        switch (type)
        {
        case LightType::Directional:
            return "Directional";
        case LightType::Spot:
            return "Spot";
        case LightType::Point:
            return "Point";
        case LightType::None:
            return "None";
        }
    }

    constexpr auto ShadowTypeToString(ShadowType type)
    {
        switch (type)
        {
        case ShadowType::Hard:
            return "Hard";
        case ShadowType::Soft:
            return "Soft";
        case ShadowType::None:
            return "None";
        }
    }

    constexpr auto ShadowTypeToString(ShadowMapResolution resolution)
    {
        switch (resolution)
        {
        case ShadowMapResolution::Low512x512:
            return "Low512x512";
        case ShadowMapResolution::Medium1024x1024:
            return "Medium1024x1024";
        case ShadowMapResolution::High2048x2048:
            return "High2048x2048";
        case ShadowMapResolution::VeryHigh4096x4096:
            return "VeryHigh4096x4096";
        }
    }
}

#endif