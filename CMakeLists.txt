cmake_minimum_required(VERSION 3.10)
project(jogo_psel_fog C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCES
    src/main.c
    src/glad.c
    src/stb_vorbis_impl.c
)

find_package(glfw3 CONFIG REQUIRED)

find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

if(MINGW)
    target_link_options(jogo_psel_fog PRIVATE -static -static-libgcc -static-libstdc++)
    set(GLFW_BUILD_DLL OFF CACHE BOOL "" FORCE)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE STBI_NO_SIMD)

target_link_libraries(${PROJECT_NAME}
    glfw
    ${OPENGL_LIBRARIES}
    m
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} m)
endif()