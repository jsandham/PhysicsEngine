@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

set SRC="../src"

set ENGINE_INCLUDE="../../engine/include"
set GLEW_INCLUDE="../../engine/include/glew-2.1.0"
set FREETYPE_INCLUDE="../../engine/include/freetype"

 :: static library
set ENGINE_LIB="../../engine/lib/debug/engine.lib"

:: import libraries (make sure dll's is available)
set GLEW_LIB="../../engine/lib/debug/glew32.lib" 
set FREETYPE_LIB="../../engine/lib/debug/freetype.lib"

:: import libraries (dll's are available by system)
set STANDARD_LIB="opengl32.lib" "user32.lib" "ole32.lib"

cl /Fe /I%ENGINE_INCLUDE% /I%GLEW_INCLUDE% /I%FREETYPE_INCLUDE% /O2 /MDd -Zi %SRC%/main.cpp %SRC%/Load.cpp %SRC%/UnitTests.cpp %ENGINE_LIB% %GLEW_LIB% %FREETYPE_LIB% %STANDARD_LIB%