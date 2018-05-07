@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW= "C:\Program Files (x86)\glew-2.1.0\include"

set CompilerFlags= -Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -Zi /I%GLEW%
set LinkerFlags= user32.lib gdi32.lib xinput.lib opengl32.lib glew32.lib 

set Core= ..\src\core\Octtree.cpp ..\src\core\Physics.cpp ..\src\core\Log.cpp ..\src\core\Input.cpp ..\src\core\Time.cpp ..\src\core\Sphere.cpp ..\src\core\Bounds.cpp ..\src\core\Ray.cpp ..\src\core\Line.cpp ..\src\core\Capsule.cpp ..\src\core\Geometry.cpp ..\src\core\Frustum.cpp  ..\src\core\Mesh.cpp

set Asset= ..\src\MeshLoader.cpp ..\src\TextureLoader.cpp

set Entities= ..\src\entities\Entity.cpp

set Components= ..\src\components\Component.cpp ..\src\components\Camera.cpp ..\src\components\Collider.cpp ..\src\components\BoxCollider.cpp ..\src\components\SphereCollider.cpp ..\src\components\CapsuleCollider.cpp ..\src\components\Transform.cpp ..\src\components\Rigidbody.cpp ..\src\components\DirectionalLight.cpp ..\src\components\SpotLight.cpp ..\src\components\PointLight.cpp ..\src\components\MeshRenderer.cpp ..\src\components\LineRenderer.cpp

set Systems= ..\src\systems\System.cpp ..\src\systems\RenderSystem.cpp ..\src\systems\PhysicsSystem.cpp ..\src\systems\PlayerSystem.cpp ..\src\systems\DebugSystem.cpp ..\src\systems\CleanUpSystem.cpp

set Graphics= ..\src\graphics\Buffer.cpp ..\src\graphics\Color.cpp ..\src\graphics\Texture.cpp ..\src\graphics\Cubemap.cpp ..\src\graphics\Texture2D.cpp ..\src\graphics\Texture3D.cpp ..\src\graphics\Framebuffer.cpp ..\src\graphics\Gizmos.cpp ..\src\graphics\GraphicState.cpp ..\src\graphics\Material.cpp ..\src\graphics\OpenGL.cpp ..\src\graphics\Shader.cpp ..\src\graphics\ShaderUniformState.cpp ..\src\graphics\UniformBufferObject.cpp ..\src\graphics\VertexArrayObject.cpp ..\src\graphics\stb_image_implementation.cpp

set Memory= ..\src\memory\Manager.cpp ..\src\memory\Pool.cpp 

set Win32= ..\src\platform\Win32\win32_main.cpp

mkdir build
pushd build
:: build win32
cl %CompilerFlags% %Win32% ..\src\Scene.cpp %Core% %Asset% %Entities% %Components% %Systems% %Graphics% %Memory% %LinkerFlags%

popd




