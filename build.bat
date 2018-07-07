@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -Zi
set Libs=user32.lib gdi32.lib xinput.lib opengl32.lib glew32.lib cudart.lib
set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64

set Core=..\src\core\Octtree.cpp ..\src\core\Physics.cpp ..\src\core\Log.cpp ..\src\core\Input.cpp ..\src\core\Time.cpp ..\src\core\Sphere.cpp ..\src\core\Bounds.cpp ..\src\core\Ray.cpp ..\src\core\Line.cpp ..\src\core\Capsule.cpp ..\src\core\Geometry.cpp ..\src\core\Frustum.cpp  ..\src\core\GMesh.cpp ..\src\core\Mesh.cpp 
set Asset=..\src\MeshLoader.cpp ..\src\TextureLoader.cpp
set Entities=..\src\entities\Entity.cpp
set Components=..\src\components\Component.cpp ..\src\components\Solid.cpp ..\src\components\Cloth.cpp ..\src\components\Fluid.cpp ..\src\components\Joint.cpp ..\src\components\HingeJoint.cpp ..\src\components\SpringJoint.cpp ..\src\components\Camera.cpp ..\src\components\Collider.cpp ..\src\components\BoxCollider.cpp ..\src\components\SphereCollider.cpp ..\src\components\CapsuleCollider.cpp ..\src\components\Transform.cpp ..\src\components\Rigidbody.cpp ..\src\components\DirectionalLight.cpp ..\src\components\SpotLight.cpp ..\src\components\PointLight.cpp ..\src\components\MeshRenderer.cpp ..\src\components\LineRenderer.cpp
set Systems=..\src\systems\System.cpp ..\src\systems\RenderSystem.cpp ..\src\systems\PhysicsSystem.cpp ..\src\systems\PlayerSystem.cpp ..\src\systems\DebugSystem.cpp ..\src\systems\CleanUpSystem.cpp
set Graphics=..\src\graphics\Buffer.cpp ..\src\graphics\Color.cpp ..\src\graphics\Texture.cpp ..\src\graphics\Cubemap.cpp ..\src\graphics\Texture2D.cpp ..\src\graphics\Texture3D.cpp ..\src\graphics\Framebuffer.cpp ..\src\graphics\Gizmos.cpp ..\src\graphics\GraphicState.cpp ..\src\graphics\Material.cpp ..\src\graphics\OpenGL.cpp ..\src\graphics\Shader.cpp ..\src\graphics\ShaderUniformState.cpp ..\src\graphics\UniformBufferObject.cpp ..\src\graphics\VertexArrayObject.cpp ..\src\graphics\stb_image_implementation.cpp
set Memory=..\src\memory\Manager.cpp ..\src\memory\Pool.cpp 
set Win32=..\src\platform\Win32\win32_main.cpp
set Cuda=..\src\cuda\fluid_kernels.cu ..\src\cuda\cloth_kernels.cu ..\src\cuda\solid_kernels.cu ..\src\cuda\QuadratureRule.cu
set Solvers=..\src\solvers\AMG.cpp ..\src\solvers\SLAF.cpp ..\src\solvers\debug.cpp

set CoreObj=Octtree.obj Physics.obj Log.obj Input.obj Time.obj Sphere.obj Bounds.obj Ray.obj Line.obj Capsule.obj Geometry.obj Frustum.obj GMesh.obj Mesh.obj
set AssetObj=MeshLoader.obj TextureLoader.obj
set EntitiesObj=Entity.obj
set ComponentsObj=Component.obj Solid.obj Cloth.obj Fluid.obj Joint.obj HingeJoint.obj SpringJoint.obj Camera.obj Collider.obj BoxCollider.obj SphereCollider.obj CapsuleCollider.obj Transform.obj Rigidbody.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj LineRenderer.obj
set SystemsObj=System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj DebugSystem.obj CleanUpSystem.obj
set GraphicsObj= Buffer.obj Color.obj VertexArrayObject.obj UniformBufferObject.obj Texture.obj Cubemap.obj Texture2D.obj Texture3D.obj Framebuffer.obj OpenGL.obj GraphicState.obj Shader.obj ShaderUniformState.obj Material.obj stb_image_implementation.obj Gizmos.obj
set MemoryObj=Manager.obj Pool.obj
set Win32Obj=win32_main.obj
set CudaObj=fluid_kernels.obj cloth_kernels.obj solid_kernels.obj CudaPhysics.obj QuadratureRule.obj
set SolversObj=AMG.obj SLAF.obj debug.obj


mkdir build
pushd build

:: compile cuda source files to obj
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\fluid_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\cloth_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\CudaPhysics.cu

:: compile c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% %Win32% ..\src\Scene.cpp ..\src\cuda\Util.cpp %Core% %Asset% %Entities% %Components% %Systems% %Graphics% %Memory% %Solvers%

:: link 
link %LinkerFlags% %CoreObj% %AssetObj% %EntitiesObj% %ComponentsObj% %SystemsObj% %GraphicsObj% %MemoryObj% %Win32Obj% %CudaObj% %SolversObj% Scene.obj Util.obj %Libs%
rem link %LinkerFlags% %Win32Obj% %CoreObj% %EntitiesObj% %ComponentsObj% Buffer.obj Color.obj VertexArrayObject.obj UniformBufferObject.obj Texture.obj Cubemap.obj Texture2D.obj Texture3D.obj Framebuffer.obj OpenGL.obj GraphicState.obj Shader.obj ShaderUniformState.obj Material.obj MeshLoader.obj TextureLoader.obj Gizmos.obj stb_image_implementation.obj Manager.obj Pool.obj System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj DebugSystem.obj CleanUpSystem.obj Scene.obj Util.obj fluid_kernels.obj cloth_kernels.obj CudaPhysics.obj %Libs% 

popd




