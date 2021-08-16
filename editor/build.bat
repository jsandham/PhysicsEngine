@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

::set ENGINE_INC="../engine/include"
::set YAML_INC="../external/yaml-cpp/include"
::set GLEW_INC="../external/glew-2.1.0"
::set FREETYPE_INC="../external/freetype"
set ENGINE_INC="../../engine/include"
set YAML_INC="../../external/yaml-cpp/include"
set GLEW_INC="../../external/glew-2.1.0"
set FREETYPE_INC="../../external/freetype"

set INCLUDES=/I%ENGINE_INC% /I%YAML_INC% /I%GLEW_INC% /I%FREETYPE_INC%

::set ENGINE_LIB="../engine/lib/debug/engine.lib"
::set YAML_LIB="../external/yaml-cpp/build/Debug/yaml-cppd.lib"
::set GLEW_LIB="../engine/lib/debug/glew32.lib"
::set FREETYPE_LIB="../engine/lib/debug/freetype.lib"
set ENGINE_LIB="../../engine/lib/debug/engine.lib"
set YAML_LIB="../../external/yaml-cpp/build/Debug/yaml-cppd.lib"
set GLEW_LIB="../../engine/lib/debug/glew32.lib"
set FREETYPE_LIB="../../engine/lib/debug/freetype.lib"

set LIBS=kernel32.lib user32.lib gdi32.lib ole32.lib opengl32.lib %ENGINE_LIB% %YAML_LIB% %GLEW_LIB% %FREETYPE_LIB%

set OPT=/Od
set WARN=-W4 -wd4100 -wd4996 -wd4211
set FLAGS=/MDd -Zi -nologo /EHsc

set SOURCE=%1
set EXECUTABLE=%2
set COMPILER="C:\\Program Files\\LLVM\\bin\\clang-cl"

echo %SOURCE%
echo %EXECUTABLE%

call %COMPILER% -o %EXECUTABLE% %SOURCE% %INCLUDES% %OPT% %WARN% %FLAGS% %LIBS%

PAUSE