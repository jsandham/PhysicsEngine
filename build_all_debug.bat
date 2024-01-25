@echo off

if not defined DevEnvDir (
	call "shell.bat"
)

echo [94mBuilding imgui...[0m
cd "%~dp0\external\imgui\lib"
call "build_imgui.bat" /debug /O2
cd "..\..\.."

echo [94mBuilding imguizmo...[0m
cd "%~dp0\external\imguizmo\lib"
call "build_imguizmo.bat" /debug /O2
cd "..\..\.."

echo [94mBuilding imguicolortextedit...[0m
cd "%~dp0\external\imguicolortextedit\lib"
call "build_imguicolortextedit.bat" /debug /O2
cd "..\..\.."

echo [94mBuilding efsw...[0m
cd "%~dp0\external\efsw\lib"
call "build_efsw.bat" /debug /O2
cd "..\..\.."

echo [94mBuilding yaml-cpp...[0m
cd "%~dp0\external\yaml-cpp\lib"
call "build-yaml-cpp.bat" /debug /O2
cd "..\..\.."

echo [94mBuilding shader_cpp_generator...[0m
cd "%~dp0\engine\tools\shader_cpp_generator"
call "build_shader_cpp_generator.bat" /debug /O2
cd "..\..\.."

echo [95mGenerating cpp shaders...[0m
cd "%~dp0\engine\tools\shader_cpp_generator"
call "shader_cpp_generator.exe"
call "shader_cpp_generator_hlsl.exe"
cd "..\..\.."

echo [94mBuilding engine...[0m
cd "%~dp0\engine\lib"
call "build_engine.bat" /debug /omp /O2
cd "..\.."

echo [94mBuilding editor...[0m
cd "%~dp0\editor\bin"
call "build_editor.bat" /debug /O2
cd "..\.."