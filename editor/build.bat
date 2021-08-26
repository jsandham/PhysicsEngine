@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

set ENGINE_INC="../../../engine/include"
set YAML_INC="../../../external/yaml-cpp/include"
set GLEW_INC="../../../external/glew-2.1.0"
set FREETYPE_INC="../../../external/freetype"
set FREETYPE_INC="../../../external/glm"

set INCLUDES=/I%ENGINE_INC% /I%YAML_INC% /I%GLEW_INC% /I%FREETYPE_INC% /I%GLM%

set ENGINE_LIB="../../../engine/lib/debug/engine.lib"
set YAML_LIB="../../../external/yaml-cpp/build/Debug/yaml-cppd.lib"
set GLEW_LIB="../../../engine/lib/debug/glew32.lib"
set FREETYPE_LIB="../../../engine/lib/debug/freetype.lib"

set LIBS=kernel32.lib user32.lib gdi32.lib ole32.lib opengl32.lib %ENGINE_LIB% %YAML_LIB% %GLEW_LIB% %FREETYPE_LIB%

set OPT=/Od
set WARN=-W4 -wd4100 -wd4996 -wd4211
set FLAGS=/MDd -Zi -nologo /EHsc

set INCLUDE_PATH=%1
set SOURCE_PATH=%2
set EXECUTABLE=%3
set COMPILER="C:\\Program Files\\LLVM\\bin\\clang-cl"

echo %INCLUDE_PATH%
echo %SOURCE_PATH%
echo %EXECUTABLE%

set INCLUDE_FILES=
for /r %INCLUDE_PATH% %%v in (*.h) do (
	call :concat_inc %%v
	echo %%v
)

set SRC_FILES=
for /r %SOURCE_PATH% %%v in (*.cpp) do (
	call :concat_src %%v
	echo %%v
)

::echo %INCLUDE_FILES%
::echo %SRC_FILES%

call %COMPILER% -o %EXECUTABLE% %SRC_FILES% %INCLUDE_FILES% %INCLUDES% %OPT% %WARN% %FLAGS% %LIBS%

PAUSE




goto :eof
:concat_inc
set INCLUDE_FILES=%INCLUDE_FILES% %1
goto :eof

goto :eof
:concat_src
set SRC_FILES=%SRC_FILES% %1
goto :eof

::echo [92mCompiling game...[0m
::for /R "../src/" %%f in (*.cpp) do (
::	call %COMPILER% -o /I%INCLUDES% %OPT% %WARN% %FLAGS% %%f
::)