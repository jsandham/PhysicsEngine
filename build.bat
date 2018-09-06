@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi -nologo
set Libs=user32.lib gdi32.lib xinput.lib opengl32.lib glew32.lib cudart.lib
set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:engine.exe /DEBUG

rem set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi -nologo -LD -FS
rem set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /DLL /OUT:engine.dll

mkdir build
pushd build

:: compile engine cuda source files to obj
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\engine\src\cuda\fluid_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\engine\src\cuda\cloth_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\engine\src\cuda\CudaPhysics.cu

:: compile engine core c++ source files
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\stb_image\stb_image_implementation.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Scene.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\SceneContext.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\SceneManager.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\SceneSettings.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Manager.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Entity.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Octtree.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Physics.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Log.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Time.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Sphere.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Bounds.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Ray.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Line.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Capsule.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Geometry.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Frustum.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\GMesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Mesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Material.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Color.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Texture.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Texture2D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Texture3D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Cubemap.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\core\Shader.cpp

:: compile engine components c++ source files
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Component.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Solid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Cloth.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Fluid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Joint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\HingeJoint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\SpringJoint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Camera.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Collider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\BoxCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\SphereCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\CapsuleCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Transform.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\Rigidbody.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\DirectionalLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\SpotLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\PointLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\components\MeshRenderer.cpp

:: compile engine systems c++ source files
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\systems\System.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\systems\RenderSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\systems\PhysicsSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\systems\PlayerSystem.cpp

:: compile engine graphics c++ source files
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\Buffer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\Framebuffer.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\GraphicState.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\OpenGL.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\Graphics.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\Shader.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\ShaderUniformState.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\UniformBufferObject.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\graphics\VertexArrayObject.cpp


:: compile engine platform c++ source files
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\engine\src\platform\Win32\win32_main.cpp

rem lib -nologo -out:engine.lib stb_image_implementation.obj SceneContext.obj SceneManager.obj Manager.obj Entity.obj Octtree.obj Physics.obj Log.obj Input.obj Time.obj Sphere.obj Bounds.obj Ray.obj Line.obj Capsule.obj Geometry.obj Frustum.obj GMesh.obj Mesh.obj Material.obj Color.obj Texture.obj Texture2D.obj Texture3D.obj Cubemap.obj Shader.obj Component.obj Solid.obj Cloth.obj Fluid.obj Joint.obj HingeJoint.obj SpringJoint.obj Camera.obj Collider.obj BoxCollider.obj SphereCollider.obj CapsuleCollider.obj Transform.obj Rigidbody.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj win32_main.obj Graphics.obj Buffer.obj OpenGL.obj Util.obj fluid_kernels.obj cloth_kernels.obj solid_kernels.obj math_kernels.obj jacobi_kernels.obj pcg_kernels.obj CudaPhysics.obj CudaSolvers.obj

:: link 
link %LinkerFlags% stb_image_implementation.obj SceneContext.obj SceneManager.obj Manager.obj Entity.obj Octtree.obj Physics.obj Log.obj Input.obj Time.obj Sphere.obj Bounds.obj Ray.obj Line.obj Capsule.obj Geometry.obj Frustum.obj GMesh.obj Mesh.obj Material.obj Color.obj Texture.obj Texture2D.obj Texture3D.obj Cubemap.obj Shader.obj Component.obj Solid.obj Cloth.obj Fluid.obj Joint.obj HingeJoint.obj SpringJoint.obj Camera.obj Collider.obj BoxCollider.obj SphereCollider.obj CapsuleCollider.obj Transform.obj Rigidbody.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj win32_main.obj Graphics.obj Buffer.obj OpenGL.obj Util.obj fluid_kernels.obj cloth_kernels.obj solid_kernels.obj math_kernels.obj jacobi_kernels.obj pcg_kernels.obj CudaPhysics.obj CudaSolvers.obj %Libs%

popd











:: compile engine c++ source files to obj
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% %Win32% ..\src\cuda\Util.cpp %Core% %Entities% %Components% %Systems% %Graphics% %Solvers%
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% %Win32% ..\src\cuda\Util.cpp %Core% %Components% %Systems% %External%

:: link 
rem link %LinkerFlags% %CoreObj% %EntitiesObj% %ComponentsObj% %SystemsObj% %GraphicsObj% %Win32Obj% %CudaObj% %SolversObj% Util.obj %Libs%
rem link %LinkerFlags% %CoreObj% %ComponentsObj% %SystemsObj% %Win32Obj% %ExternalObj% Util.obj %Libs%

rem link %LinkerFlags% %Win32Obj% %CoreObj% %EntitiesObj% %ComponentsObj% Buffer.obj Color.obj VertexArrayObject.obj UniformBufferObject.obj Texture.obj Cubemap.obj Texture2D.obj Texture3D.obj Framebuffer.obj OpenGL.obj GraphicState.obj Shader.obj ShaderUniformState.obj Material.obj MeshLoader.obj TextureLoader.obj Gizmos.obj stb_image_implementation.obj Manager.obj Pool.obj System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj DebugSystem.obj CleanUpSystem.obj Scene.obj Util.obj fluid_kernels.obj cloth_kernels.obj CudaPhysics.obj %Libs% 


rem set External=..\src\stb_image\stb_image_implementation.cpp
rem set Core=..\src\core\Scene.cpp ..\src\core\SceneSettings.cpp ..\src\core\Manager.cpp ..\src\core\Entity.cpp ..\src\core\Octtree.cpp ..\src\core\Physics.cpp ..\src\core\Log.cpp ..\src\core\Input.cpp ..\src\core\Time.cpp ..\src\core\Sphere.cpp ..\src\core\Bounds.cpp ..\src\core\Ray.cpp ..\src\core\Line.cpp ..\src\core\Capsule.cpp ..\src\core\Geometry.cpp ..\src\core\Frustum.cpp  ..\src\core\GMesh.cpp ..\src\core\Mesh.cpp ..\src\core\Material.cpp ..\src\core\Color.cpp ..\src\core\Texture.cpp ..\src\core\Texture2D.cpp ..\src\core\Shader.cpp
rem set Asset=..\src\MeshLoader.cpp ..\src\TextureLoader.cpp
rem set Entities=..\src\entities\Entity.cpp
rem set Components=..\src\components\Component.cpp ..\src\components\Solid.cpp ..\src\components\Cloth.cpp ..\src\components\Fluid.cpp ..\src\components\Joint.cpp ..\src\components\HingeJoint.cpp ..\src\components\SpringJoint.cpp ..\src\components\Camera.cpp ..\src\components\Collider.cpp ..\src\components\BoxCollider.cpp ..\src\components\SphereCollider.cpp ..\src\components\CapsuleCollider.cpp ..\src\components\Transform.cpp ..\src\components\Rigidbody.cpp ..\src\components\DirectionalLight.cpp ..\src\components\SpotLight.cpp ..\src\components\PointLight.cpp ..\src\components\MeshRenderer.cpp
rem set Systems=..\src\systems\System.cpp ..\src\systems\RenderSystem.cpp ..\src\systems\PhysicsSystem.cpp ..\src\systems\PlayerSystem.cpp
rem set Graphics=..\src\graphics\Buffer.cpp ..\src\graphics\Color.cpp ..\src\graphics\Texture.cpp ..\src\graphics\Cubemap.cpp ..\src\graphics\Texture2D.cpp ..\src\graphics\Texture3D.cpp ..\src\graphics\Framebuffer.cpp ..\src\graphics\GraphicState.cpp ..\src\graphics\Material.cpp ..\src\graphics\OpenGL.cpp ..\src\graphics\Shader.cpp ..\src\graphics\ShaderUniformState.cpp ..\src\graphics\UniformBufferObject.cpp ..\src\graphics\VertexArrayObject.cpp ..\src\graphics\stb_image_implementation.cpp
rem set Memory=..\src\memory\Manager.cpp ..\src\memory\Pool.cpp 
rem set Win32=..\src\platform\Win32\win32_main.cpp
rem set Cuda=..\src\cuda\fluid_kernels.cu ..\src\cuda\cloth_kernels.cu ..\src\cuda\solid_kernels.cu ..\src\cuda\math_kernels.cu ..\src\cuda\jacobi_kernels.cu ..\src\cuda\pcg_kernels.cu ..\src\cuda\CudaPhysics.cu ..\src\cuda\CudaSolvers.cu
rem set Solvers=..\src\solvers\AMG.cpp ..\src\solvers\SLAF.cpp ..\src\solvers\debug.cpp

rem set ExternalObj=stb_image_implementation.obj
rem set CoreObj=Scene.obj SceneSettings.obj Manager.obj Entity.obj Octtree.obj Physics.obj Log.obj Input.obj Time.obj Sphere.obj Bounds.obj Ray.obj Line.obj Capsule.obj Geometry.obj Frustum.obj GMesh.obj Mesh.obj Material.obj Color.obj Texture.obj Texture2D.obj Shader.obj
rem set AssetObj=MeshLoader.obj TextureLoader.obj
rem set EntitiesObj=Entity.obj
rem set ComponentsObj=Component.obj Solid.obj Cloth.obj Fluid.obj Joint.obj HingeJoint.obj SpringJoint.obj Camera.obj Collider.obj BoxCollider.obj SphereCollider.obj CapsuleCollider.obj Transform.obj Rigidbody.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj
rem set SystemsObj=System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj
rem set GraphicsObj= Buffer.obj Color.obj VertexArrayObject.obj UniformBufferObject.obj Texture.obj Cubemap.obj Texture2D.obj Texture3D.obj Framebuffer.obj OpenGL.obj GraphicState.obj Shader.obj ShaderUniformState.obj Material.obj stb_image_implementation.obj
rem set MemoryObj=Manager.obj Pool.obj
rem set Win32Obj=win32_main.obj
rem set CudaObj=fluid_kernels.obj cloth_kernels.obj solid_kernels.obj math_kernels.obj jacobi_kernels.obj pcg_kernels.obj CudaPhysics.obj CudaSolvers.obj
rem set SolversObj=AMG.obj SLAF.obj debug.obj
