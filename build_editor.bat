@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -Zi
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64

set Files=..\src\editor\MeshLoader.cpp ..\src\editor\TextureLoader.cpp ..\src\entities\Entity.cpp ..\src\core\SceneSettings.cpp ..\src\core\Log.cpp ..\src\core\Input.cpp ..\src\core\Time.cpp ..\src\core\Frustum.cpp ..\src\core\Manager.cpp ..\src\core\Scene.cpp ..\src\core\Material.cpp ..\src\core\Mesh.cpp ..\src\core\GMesh.cpp ..\src\editor\stb_image_implementation.cpp ..\src\core\Color.cpp ..\src\core\Texture.cpp ..\src\core\Texture2D.cpp
set Components=..\src\components\Component.cpp ..\src\components\Transform.cpp ..\src\components\Rigidbody.cpp ..\src\components\Camera.cpp ..\src\components\DirectionalLight.cpp ..\src\components\SpotLight.cpp ..\src\components\PointLight.cpp ..\src\components\MeshRenderer.cpp

set FilesObj=MeshLoader.obj TextureLoader.obj Entity.obj SceneSettings.obj Log.obj Input.obj Time.obj Manager.obj Scene.Obj Frustum.obj Material.obj Mesh.obj GMesh.obj stb_image_implementation.obj Color.obj Texture.obj Texture2D.obj
set ComponentsObj=Component.obj Transform.obj Rigidbody.obj Camera.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj

mkdir build
pushd build

:: compile editor c++ files to obj
cl /c %CompilerFlags% %Files% %Components% ..\src\editor\main.cpp

:: link 
link %LinkerFlags% %FilesObj% %ComponentsObj% main.obj

popd