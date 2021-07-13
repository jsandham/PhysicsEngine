#ifndef __INTERACTION_TYPE_H__
#define __INTERACTION_TYPE_H__

namespace PhysicsEditor
{
    enum class InteractionType
    {
        None,
        Entity,
        Texture2D,
        Texture3D,
        Cubemap,
        Shader,
        Material,
        Mesh,
        Font,
        Sprite,
        CodeFile,
        File,
        Folder
    };
}

#endif