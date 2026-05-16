cmake_minimum_required(VERSION 3.10)
project(jogo_psel_fog C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCES
    src/main.c
    src/glad.c
)

# Encontra GLFW usando o modo CONFIG (mais portável)
find_package(glfw3 CONFIG REQUIRED)

# Encontra OpenGL
find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

# Desativa SIMD no stb_image.h para Windows
target_compile_definitions(${PROJECT_NAME} PRIVATE STBI_NO_SIMD)

# Linka as bibliotecas
target_link_libraries(${PROJECT_NAME}
    glfw
    ${OPENGL_LIBRARIES}
)

# No Linux, linka a biblioteca matemática (no Windows já está na libc)
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} m)
endif()