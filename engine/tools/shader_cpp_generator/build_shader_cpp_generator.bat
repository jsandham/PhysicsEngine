@echo off

call cl /nologo /FS /MDd -Zi /EHsc /std:c++17 /Fe shader_cpp_generator.cpp
call cl /nologo /FS /MDd -Zi /EHsc /std:c++17 /Fe shader_cpp_generator_hlsl.cpp