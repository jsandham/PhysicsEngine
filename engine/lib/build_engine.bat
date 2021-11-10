@echo off

set GLEW="../../external/glew-2.1.0"
set FREETYPE="../../external/freetype"
set YAML="../../external/yaml-cpp/include"
set GLM="../../external/glm"
set TINY_OBJ="../../external/tinyobjloader"
set STB="../../external/stb"
set WARN=-W4 -wd4100 -wd4996 -wd4211 -wd4201
::set WARN=-W4 -Wno-pessimizing-move -Wno-unused-parameter
set OPENMP=
set OPT=/Od
set MODEFLAGS=/FS /MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
set MODE=debug
set FLAGS=-nologo /EHsc

:: run through batch file parameter inputs
for %%x in (%*) do (
	if "%%x"=="/help" (
		echo "help"
	)
	if "%%x"=="/omp" (
		set OPENMP=/openmp
	)
	if "%%x"=="/O2" (
		set OPT=/O2
	)
	if "%%x"=="/debug" (
		set MODE=debug
		set MODEFLAGS=/FS /MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/FS /MD /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
if defined OPENMP (
	echo [92mOpenMP: on[0m	
)else (
	echo [92mOpenMP: off[0m	
)
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code
echo [92mCompiling C++ engine code...[0m
for /R "../src/" %%f in (*.cpp) do (
	call cl /c /std:c++17 /I%GLEW% /I%FREETYPE% /I%YAML% /I%GLM% /I%TINY_OBJ% /I%STB% %OPT% %OPENMP% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
)

:: create static engine library
echo [92mCreating static engine library...[0m
lib /nologo /out:%MODE%/engine.lib %OBJ_FILES%

:: delete .obj fles
::echo [92mDeleting objects...[0m
::set OBJ_FILES=
::for /r "%MODE%/obj" %%v in (*.obj) do (
::	del /s %%v
::)

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof