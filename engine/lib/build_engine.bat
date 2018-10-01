@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi -nologo

:: compile engine cuda source files to obj
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\fluid_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\cloth_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\solid_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\math_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\jacobi_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\pcg_kernels.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\CudaPhysics.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64" -I%CUDA% -I%GLEW% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" --compile ..\src\cuda\CudaSolvers.cu

:: compile engine core c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\stb_image\stb_image_implementation.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\SceneContext.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\SceneManager.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Manager.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Entity.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Octtree.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Physics.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Log.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Input.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Time.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Sphere.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Bounds.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Ray.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Line.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Capsule.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Geometry.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Frustum.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\GMesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Mesh.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Material.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Color.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture2D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Texture3D.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Cubemap.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Shader.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\core\Util.cpp

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
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\CapsuleCollider.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Transform.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\Rigidbody.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\DirectionalLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\SpotLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\PointLight.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\components\MeshRenderer.cpp

:: compile engine systems c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\LoadInternalSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\System.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\RenderSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\PhysicsSystem.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\systems\PlayerSystem.cpp

:: compile engine graphics c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\Buffer.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\Framebuffer.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\GraphicState.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\OpenGL.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\Graphics.cpp
rem cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\Shader.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\ShaderUniformState.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\UniformBufferObject.cpp
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\graphics\VertexArrayObject.cpp

:: compile engine platform c++ source files to obj
cl /c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" /I%GLEW% %CompilerFlags% ..\src\platform\Win32\win32_main.cpp

:: create lib library
set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi -nologo -LD -FS
lib -nologo -out:engine.lib stb_image_implementation.obj SceneContext.obj SceneManager.obj Manager.obj Entity.obj Octtree.obj Physics.obj Log.obj Input.obj Time.obj Sphere.obj Bounds.obj Ray.obj Line.obj Capsule.obj Geometry.obj Frustum.obj GMesh.obj Mesh.obj Material.obj Color.obj Texture.obj Texture2D.obj Texture3D.obj Cubemap.obj Shader.obj Component.obj Solid.obj Cloth.obj Fluid.obj Joint.obj HingeJoint.obj SpringJoint.obj Camera.obj Collider.obj BoxCollider.obj SphereCollider.obj CapsuleCollider.obj Transform.obj Rigidbody.obj DirectionalLight.obj SpotLight.obj PointLight.obj MeshRenderer.obj LoadInternalSystem.obj System.obj RenderSystem.obj PhysicsSystem.obj PlayerSystem.obj win32_main.obj Graphics.obj Buffer.obj OpenGL.obj Util.obj fluid_kernels.obj cloth_kernels.obj solid_kernels.obj math_kernels.obj jacobi_kernels.obj pcg_kernels.obj CudaPhysics.obj CudaSolvers.obj

popd