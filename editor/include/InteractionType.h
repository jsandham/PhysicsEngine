#ifndef INTERACTION_TYPE_H__
#define INTERACTION_TYPE_H__

namespace PhysicsEditor
{
	enum class InteractionType
	{
		None,
		Scene,
		Entity,
		Texture2D,
		Texture3D,
		Cubemap,
		Shader,
		Material,
		Mesh,
		Font,
		Sprite,
		RenderTexture,
		CodeFile,
		File,
		Folder
	};
}

#endif