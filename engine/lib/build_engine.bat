@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

set GLEW="../include/glew-2.1.0"
set FREETYPE="../include/freetype"
set WARN=-W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -wd4512 -wd4458 
set OPENMP=
set OPT=/Od
set MODEFLAGS=/MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
set MODE=debug
set FLAGS=-nologo

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
		set MODEFLAGS=/MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/MD /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mOptimization level: %MODE%[0m
if defined OPENMP (
	echo [92mOpenMP: on[0m	
)else (
	echo [92mOpenMP: off[0m	
)
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code
echo [92mCompiling C++ engine code...[0m
for /R "../src/" %%f in (*.cpp) do (
	call cl /c /I%GLEW% /I%FREETYPE% %OPT% %OPENMP% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
	echo %%v
)

:: create static lib library
echo [92mCreate static engine library[0m
lib -out:%MODE%/engine.lib %OBJ_FILES%

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof