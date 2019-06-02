@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
set FREETYPE="C:\Users\James\Documents\PhysicsEngine\engine\include\freetype"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -wd4512 -Zi -nologo /Fo"obj"\ /Fd"obj"\

:: compile engine cuda source files to obj
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\boids_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\fluid_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\cloth_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\solid_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\math_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\jacobi_kernels.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\kernels\pcg_kernels.cu

rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\CudaPhysics.cu
rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\CudaSolvers.cu

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\BoidsDeviceData.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\ClothDeviceData.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\FluidDeviceData.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile --output-directory obj\ ..\src\cuda\SolidDeviceData.cu

:: compile engine core c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\stb_image\stb_image_implementation.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\LoadInternal.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\SceneContext.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\AssetLoader.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\AssetDatabase.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\WorldManager.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\World.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\SceneGraph.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Asset.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Entity.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Octtree.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\UniformGrid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\HGrid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Physics.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Log.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Input.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Time.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Sphere.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Bounds.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Triangle.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Ray.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Line.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Capsule.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Geometry.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Frustum.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\GMesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Mesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Material.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Color.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture2D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture3D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Cubemap.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Shader.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% /I%FREETYPE% %CompilerFlags% ..\src\core\Font.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Util.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Guid.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\PerformanceGraph.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\DebugWindow.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\SlabBuffer.cpp

:: compile engine components c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Component.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Solid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Cloth.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Fluid.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Joint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\HingeJoint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\SpringJoint.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Camera.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Collider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\BoxCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\SphereCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\MeshCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\CapsuleCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Transform.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Rigidbody.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Light.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\DirectionalLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\SpotLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\PointLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\MeshRenderer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\LineRenderer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Boids.cpp

:: compile engine systems c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\System.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\RenderSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\PhysicsSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\CleanUpSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\DebugSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\BoidsSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\ClothSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\FluidSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\SolidSystem.cpp

:: compile engine graphics c++ source files to obj
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\OpenGL.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\Graphics.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\BatchManager.cpp  
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\ForwardRenderer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\DeferredRenderer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\DebugRenderer.cpp

:: create lib library
set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi -nologo -LD -FS

:: create list of .obj files
set OBJ_FILES=
for /r %%v in (*.obj) do call :concat_obj %%v

lib -out:engine.lib %OBJ_FILES%

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof