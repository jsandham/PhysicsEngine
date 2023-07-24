#ifndef SERIALIZATION_ENUMS_H__
#define SERIALIZATION_ENUMS_H__

namespace PhysicsEngine
{
    enum class HideFlag
    {
        None = 0,
        DontSave = 1
    };

    constexpr auto HideFlagToString(HideFlag flag)
    {
        switch (flag)
        {
        case HideFlag::None:
            return "None";
        case HideFlag::DontSave:
            return "DontSave";
        }
    }
};

#endif