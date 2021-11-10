@echo off

if not defined DevEnvDir (
	call "shell.bat"
)

echo [94mBuilding imgui...[0m
cd "%~dp0\external\imgui\lib"
call "build_imgui.bat"
cd "..\..\.."

echo [94mBuilding imguizmo...[0m
cd "%~dp0\external\imguizmo\lib"
call "build_imguizmo.bat"
cd "..\..\.."

echo [94mBuilding simplefilewatcher...[0m
cd "%~dp0\external\simplefilewatcher\lib"
call "build_simplefilewatcher.bat"
cd "..\..\.."

echo [94mBuilding yaml-cpp...[0m
cd "%~dp0\external\yaml-cpp\lib"
call "build-yaml-cpp.bat"
cd "..\..\.."

echo [94mBuilding engine...[0m
cd "%~dp0\engine\lib"
call "build_engine.bat"
cd "..\.."