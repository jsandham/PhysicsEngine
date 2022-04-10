@echo off

if not defined DevEnvDir (
	call "shell.bat"
)

echo [94mBuilding imgui...[0m
cd "%~dp0\external\imgui\lib"
call "build_imgui.bat" /O2
cd "..\..\.."

echo [94mBuilding imguizmo...[0m
cd "%~dp0\external\imguizmo\lib"
call "build_imguizmo.bat" /O2
cd "..\..\.."

echo [94mBuilding simplefilewatcher...[0m
cd "%~dp0\external\simplefilewatcher\lib"
call "build_simplefilewatcher.bat" /O2
cd "..\..\.."

echo [94mBuilding yaml-cpp...[0m
cd "%~dp0\external\yaml-cpp\lib"
call "build-yaml-cpp.bat" /O2
cd "..\..\.."

echo [94mBuilding shader_cpp_generator...[0m
cd "%~dp0\engine\tools\shader_cpp_generator"
call "build_shader_cpp_generator.bat" /O2
cd "..\..\.."

echo [95mGenerating cpp shaders...[0m
cd "%~dp0\engine\tools\shader_cpp_generator"
call "shader_cpp_generator.exe"
cd "..\..\.."

echo [94mBuilding engine...[0m
cd "%~dp0\engine\lib"
call "build_engine.bat" /O2
cd "..\.."

echo [94mBuilding editor...[0m
cd "%~dp0\editor\bin"
call "build_editor.bat" /O2
cd "..\.."