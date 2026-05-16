set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

set(CMAKE_SYSROOT /usr/x86_64-w64-mingw32)

# Versão do GCC (ajuste conforme sua instalação)
set(GCC_VERSION "16.1.0")
set(GCC_INCLUDE "/usr/lib/gcc/x86_64-w64-mingw32/${GCC_VERSION}/include")

# Flags C: -nostdinc remove TODOS os includes padrão do sistema (host)
# Depois adicionamos explicitamente apenas os includes do target
set(CMAKE_C_FLAGS "--sysroot=${CMAKE_SYSROOT} -nostdinc -isystem ${CMAKE_SYSROOT}/include -isystem ${GCC_INCLUDE}" CACHE STRING "" FORCE)

# Flags C++ (se precisar)
set(CMAKE_CXX_FLAGS "--sysroot=${CMAKE_SYSROOT} -nostdinc++ -isystem ${CMAKE_SYSROOT}/include -isystem ${GCC_INCLUDE} -isystem ${CMAKE_SYSROOT}/include/c++/${GCC_VERSION} -isystem ${CMAKE_SYSROOT}/include/c++/${GCC_VERSION}/x86_64-w64-mingw32" CACHE STRING "" FORCE)

# Ajustes para o CMake encontrar bibliotecas
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)