cmake_minimum_required(VERSION 3.10)
project(jogo_psel_fog C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/main.c
    src/glad.c
    src/stb_image.c
)

# Find and link GLFW
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${GLFW_LIBRARIES}
    m
    GL
)

# Add include directories from pkg-config
target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_INCLUDE_DIRS})

# Add compile flags from pkg-config
target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW_CFLAGS_OTHER})
