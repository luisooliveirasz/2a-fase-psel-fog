#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

/*
.................................................................................................................
.................................................................................................................
....................................................................................................=*:..........
................................................................................................:*#*-:...........
.............................................................................................:###*--:............
..........................................................................................:+###*:-:..............
........................................................................................+**##*:-:................
.................:###*##*=:.........................................................:+***#*+:--::................
.................:*#####*#*#++=...........-+=-..-=+=++=...=-:-:..................-+***#*#=:-=--::................
..................:+*#****+***+**==----**=*=*#%*#***@**#**++**+-===+==:---=-==**######+--=+=-:::::...............
....................:=*+****+##-+--===++==-:--*#=*%@*#*+---=**-==+*-=-*=++===**#+=++=+**+=-::::..................
....................--=-===**=---==*=*:-*#=.::-%#++=+::-=#*+-*++=*+*+**=-==++=---=++#***+==:.....................
....................:=+++++-:===+=*##+-=%%#=-=+**+=#=:++%%#-=*%*==++=*+*+==--+*===--=***==-:::...................
.....................---==+*##-=-=*#*=--+%%-==*%+#++=:+%@*--=#*%+-+*#*****++-===++*+==+-==:......................
.....................:-:=*#%%*=--+##%--:=%--+*+=+-==+-+#*--:+%%%==-**%%%%##+*-++=+*+=-----:......................
....................::==+=#*##=-==#%*-::-+*-=-**+==:++++:--=+@%#+:+=%@%#***#++=++==*+=-=+=:......................
....................::--=++=+=++:-*%=--===----+++---=+++-=-==*%#-=+++**#*++++=+++++++-====--.....................
..................:=-.-::--:::.-::-++-==+=-=+--==+==+=+*=====*%*====+-*=**+-*==-=***====*+*=:....................
................:=+**=+*#####%#*-:::::--==--====-=:=+++++======-:-::-=+**##++====+**++===+*+.....................
................-###*#**%#*#*#%%@@%*-:..:-==-+=-+-++=++--::::--=*%@@@@@@%%%@@@###*#+*+*+==+=:....................
..............:-*##++::+%+++*%%#%@%@@#-..:-=----:==+++--:::=+#@@@%@%%%%%####@#**#%%%%%%#+=+*+:...................
:.............:=#==-:..:#*-=+*%%@@%#%@#:..-----=-=-==--:::=#@@%%##*@@@%%##%@*-:-=+###%%%**+**=:..................
.::...........=*=-++==-:-+%#++*@@#**#@%:.:-:::----:-==-::*%@@@@####%@@@%%+-:.:=-=++++*%%%#+***+..................
:::::........:**=++==+*+::-=+**##****#%=--::-----=--==+=#@@@%#*#***++=:----==+++***#**#%%%#****:.................
:::::::......-*+-++-:-==:+--:::::..:+%@*+=-=--=====++++*@@%*=-::::::::---=:+**++**+***#%%%##***=.................
:::::::::..::=*=:-=-:-=-=-=--=-=:.:+#%#++++=====+++*****%%#*==:.::-+==-=-+-:-=-++*++++**#%%#***=.................
:::::::::::.:=+=-:-==---=====++--=+***##+++++=++++**####+#**+:::==+--+=*+*++-=*#*****+***%@%%#*+.................
:::::::::::::-+---:=+++-+==-=:--==+*+*+**++*++++********#+#*+=-==+-++*+===*#*++++==++*+**%@@%%#*:................
::::::::::::::-----..-=*#+==+==-=*+=-+*###******#####*##*++*++++==++*+#*+**=====++-=++++*#%###%=:................
....::::::::::--=**+-=+#*+=--====----+**+********##%%#**+++=++++**++=+*%%%%#*++++++*++=++++****-.................
.............:=*%###**##=---==+---:::+***++++++++***%@%+=====+++*+++*+=**=+%@%%%%%%%#*++++=++#=:.................
..............-##*+-----==-===+*++-...:=*%#+*++#@@%%*+---====+**+****++++=+=+**#%%@@%#*+++++##=::................
..............:==-::::::.::::::::::......:#+**+*%+=---:::==+**=*##%***+====++++++*#%@%#*+***#++-:................
...............:==-::=-::-:-=-+*==-:.......*%%%#::.:::::-:---====--::=+=----:==+++*%%%%%%##%#**=::...............
................::-======:..::-:::.........:-#=.::::.:::=++*+=#***#+++===+*+++++++++*%%#*+#****+-:...............
.................:::-==+=---==-==--.-::::-:::=-::::-=::-::=--==---=-=---====+==--==+*##**#*+***+--::.::..........
..................-=-=====--:::::::.::::::--+*#==:::-=--=:----++**+++-=--===++*+++*###*++=+**##*+-:..:::........:
..................:-==-==++-::---=--:-::-=+*++=++*==----===*+++=+====-::===+*+*####**++**+*#*****+::::::.:.....:-
.................:-+==--=-====-:-----=====+-=--===*++======+*++*=+=-=---=-==+##**##**++***######*=-::::::::...::-
.................-=+=++----==:=-:::::::::---::-:-:-=----======----:----=++**=*++***+*########%##*+=::::::::::::--
.........:......:-===+*+==---=-===-::.:::::::::::::-::::::::::-:--===+*****+*#+***+#####%%####****+-::::::::::---
................:--===+==++-===---===-:::...::::::::::::::::-----==+*+*****+**+#+**#*#%%##+*##*#*++-:::::::::----
...............:-=++=-+++++++++=====+++=+-:::::::::::::-:--==+**+*##**+++++***##*%##%%#*#*######*++-::::::::-----
..............::====+=+++==+++++==-=====+*+++*=+=-=+++=**+**+****+***+*+*****#*###*#####*+#######*+-::::::::-----
...............-++*+==+==+++++=+---=-==-===+++*+******+++***++**+*++++*******###**#+*#+*+##+*#%#***+-::::::--=---
...............:++++++-+===+++=-+==--=---===++=++++**+++++++++++**++++******#%##*#*+****+###*#%####*--::::-===---
...............-+++++--======+====---==---======++++*+=+++++++++++*+*++*****####********++#**#*#%%#*=--::-=====--
..............:=+*#+++===-===-====----------=-=====+*+==++**=+++++++++******#****+*#**++++#*##%*%#**+===========-
*/

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f

const char* vertex_src =
    "#version 330 core\n"

    "layout(location = 0) in vec3 a_position;\n"
    "layout(location = 1) in vec4 a_color;\n"
    "layout(location = 2) in vec2 a_tex_coord;\n"
    "layout(location = 3) in vec3 a_normal;\n"
    "layout(location = 4) in float a_tex_index;\n"

    "uniform mat4 u_model;\n"
    "uniform mat4 u_view;\n"
    "uniform mat4 u_projection;\n"

    "out vec4 v_color;\n"
    "out vec2 v_tex_coord;\n"
    "out vec3 v_normal;\n"
    "out vec3 v_frag_pos;\n"
    "out float v_tex_index;\n"

    "void main()\n"
    "{\n"
    "    vec4 world_pos = u_model * vec4(a_position, 1.0);\n"
    "    gl_Position = u_projection * u_view * world_pos;\n"
    "    v_frag_pos  = world_pos.xyz;\n"
    "    v_normal    = mat3(transpose(inverse(u_model))) * a_normal;\n"
    "    v_color     = a_color;\n"
    "    v_tex_coord = a_tex_coord;\n"
    "    v_tex_index = a_tex_index;\n"
    "}\n";

const char* fragment_src =
    "#version 330 core\n"

    "in vec4  v_color;\n"
    "in vec2  v_tex_coord;\n"
    "in vec3  v_normal;\n"
    "in vec3  v_frag_pos;\n"
    "in float v_tex_index;\n"

    "uniform vec3        u_light_dir;\n"
    "uniform vec3        u_light_color;\n"
    "uniform sampler2D   u_textures[16];\n"
    "uniform int         u_texture_count;\n"
    "uniform vec3        u_blend_color;\n"

    "out vec4 frag_color;\n"

    "vec4 sample_texture(int idx) {\n"
    "    if (idx == 0)  return v_color;\n"
    "    if (idx == 1)  return texture(u_textures[1],  v_tex_coord);\n"
    "    if (idx == 2)  return texture(u_textures[2],  v_tex_coord);\n"
    "    if (idx == 3)  return texture(u_textures[3],  v_tex_coord);\n"
    "    if (idx == 4)  return texture(u_textures[4],  v_tex_coord);\n"
    "    if (idx == 5)  return texture(u_textures[5],  v_tex_coord);\n"
    "    if (idx == 6)  return texture(u_textures[6],  v_tex_coord);\n"
    "    if (idx == 7)  return texture(u_textures[7],  v_tex_coord);\n"
    "    if (idx == 8)  return texture(u_textures[8],  v_tex_coord);\n"
    "    if (idx == 9)  return texture(u_textures[9],  v_tex_coord);\n"
    "    if (idx == 10) return texture(u_textures[10], v_tex_coord);\n"
    "    if (idx == 11) return texture(u_textures[11], v_tex_coord);\n"
    "    if (idx == 12) return texture(u_textures[12], v_tex_coord);\n"
    "    if (idx == 13) return texture(u_textures[13], v_tex_coord);\n"
    "    if (idx == 14) return texture(u_textures[14], v_tex_coord);\n"
    "    if (idx == 15) return texture(u_textures[15], v_tex_coord);\n"
    "    return vec4(1.0, 0.0, 1.0, 1.0);\n"
    "}\n"
    
    "void main()\n"
    "{\n"
    "    float ambient_strength = 0.25;\n"
    "    vec3  ambient = ambient_strength * u_light_color;\n"

    "    vec3 norm      = normalize(v_normal);\n"
    "    float diff     = max(dot(norm, u_light_dir), 0.0);\n"
    "    vec3 diffuse   = diff * u_light_color;\n"

    "    int tex_idx = int(v_tex_index);\n"
    "    vec4 base_color = sample_texture(tex_idx);\n"

    "    vec3 final_color = base_color.rgb * (ambient + diffuse) * u_blend_color;\n"
    "    frag_color = vec4(final_color, base_color.a);\n"
    "}\n";

// ------------------------------------------------------------------
// tipos
// ------------------------------------------------------------------

typedef struct { float x, y, z;    } vec3;
typedef struct { float x, y, z, w; } vec4;
typedef struct { float x, y;       } vec2;
typedef struct { float m[16];      } mat4;

// ------------------------------------------------------------------
// constantes globais
// ------------------------------------------------------------------

vec3 MAIN_LIGHT_POS   = (vec3){ 0.0f, 20.0f, -5.0f };
vec3 MAIN_LIGHT_COLOR = (vec3){ 1.0f,  1.0f,  1.0f };
vec3 MAIN_LIGHT_DIR = (vec3){ 0.4f, 0.8f, 0.6f };

// ------------------------------------------------------------------
// declarações mat4 e vec3
// ------------------------------------------------------------------

vec3  vec3_zero();
vec3  vec3_one();
vec3  vec3_add(vec3* a, vec3* b);
vec3  vec3_subtract(vec3* a, vec3* b);
vec3  vec3_multiply_scalar(vec3* v, float s);
vec3  vec3_cross(vec3* a, vec3* b);
float vec3_dot(vec3* a, vec3* b);
vec3  vec3_lerp(vec3* a, vec3* b, float t);
float lerp(float a, float b, float t);
vec3 vec3_from_scalar(float scalar);
vec3 vec3_negate(vec3* vec);

mat4 mat4_identity();
mat4 mat4_multiply(mat4* a, mat4* b);
mat4 mat4_translate(mat4* mat, vec3* vec);
mat4 mat4_rotate_x(mat4* mat, float angle);
mat4 mat4_rotate_y(mat4* mat, float angle);
mat4 mat4_rotate_z(mat4* mat, float angle);
mat4 mat4_rotate(mat4* mat, vec3* axis, float angle);
mat4 mat4_scale(mat4* mat, vec3* vec);
mat4 mat4_perspective(float fov, float aspect, float near, float far);
mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up);

// ------------------------------------------------------------------
// math interno (renderer)
// ------------------------------------------------------------------

static void mat4_identity_raw(float out[16])
{
    memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

static void mat4_mul_raw(float out[16], float a[16], float b[16])
{
    float tmp[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                tmp[j*4+i] += a[k*4+i] * b[j*4+k];
    memcpy(out, tmp, 16 * sizeof(float));
}

static void build_translate(float out[16], vec3 pos)
{
    mat4_identity_raw(out);
    out[12] = pos.x;
    out[13] = pos.y;
    out[14] = pos.z;
}

static void build_scale(float out[16], float sx, float sy, float sz)
{
    mat4_identity_raw(out);
    out[0]  = sx;
    out[5]  = sy;
    out[10] = sz;
}

static void build_rotate_x(float out[16], float a)
{
    mat4_identity_raw(out);
    out[5]  =  cosf(a); out[9]  = -sinf(a);
    out[6]  =  sinf(a); out[10] =  cosf(a);
}

static void build_rotate_y(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0]  =  cosf(a); out[8]  =  sinf(a);
    out[2]  = -sinf(a); out[10] =  cosf(a);
}

static void build_rotate_z(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0] =  cosf(a); out[4] = -sinf(a);
    out[1] =  sinf(a); out[5] =  cosf(a);
}

static void build_transform_with_normal(float out[16],
                                        vec3 pos, vec3 rot,
                                        float sx, float sy,
                                        vec3 normal_in, vec3* normal_out)
{
    float t[16], rx[16], ry[16], rz[16], s[16], tmp[16];
    build_translate(t,  pos);
    build_rotate_x (rx, rot.x);
    build_rotate_y (ry, rot.y);
    build_rotate_z (rz, rot.z);
    build_scale    (s,  sx, sy, 1.0f);

    float rot_mat[16];
    mat4_mul_raw(rot_mat, rz, ry);
    mat4_mul_raw(rot_mat, rot_mat, rx);

    normal_out->x = rot_mat[0]*normal_in.x + rot_mat[4]*normal_in.y + rot_mat[8] *normal_in.z;
    normal_out->y = rot_mat[1]*normal_in.x + rot_mat[5]*normal_in.y + rot_mat[9] *normal_in.z;
    normal_out->z = rot_mat[2]*normal_in.x + rot_mat[6]*normal_in.y + rot_mat[10]*normal_in.z;

    // normaliza
    float len = sqrtf(normal_out->x*normal_out->x +
                      normal_out->y*normal_out->y +
                      normal_out->z*normal_out->z);
    if (len > 0.0001f) { normal_out->x /= len; normal_out->y /= len; normal_out->z /= len; }

    mat4_mul_raw(tmp, rot_mat, s);
    mat4_mul_raw(out, t, tmp);
}

static vec3 transform_vec4(float m[16], vec4 v)
{
    return (vec3){
        m[0]*v.x + m[4]*v.y + m[8] *v.z + m[12]*v.w,
        m[1]*v.x + m[5]*v.y + m[9] *v.z + m[13]*v.w,
        m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
    };
}

// ------------------------------------------------------------------
// input
// ------------------------------------------------------------------

#define MAX_KEYS 512

static GLFWwindow* m_window = NULL;
static int m_current_keys[MAX_KEYS];
static int m_previous_keys[MAX_KEYS];

void input_init(GLFWwindow* window)
{
    m_window = window;
    memset(m_current_keys,  0, sizeof(m_current_keys));
    memset(m_previous_keys, 0, sizeof(m_previous_keys));
}

void input_update()
{
    memcpy(m_previous_keys, m_current_keys, sizeof(m_current_keys));
    for (int i = 0; i < MAX_KEYS; i++)
        m_current_keys[i] = glfwGetKey(m_window, i) == GLFW_PRESS;
}

int input_get_key(int key)
{
    int pressed = glfwGetKey(m_window, key) == GLFW_PRESS;
    m_current_keys[key] = pressed;
    return pressed;
}

int input_get_key_down(int key)
{
    return  m_current_keys[key] && !m_previous_keys[key];
}

int input_get_key_up(int key)
{
    return !m_current_keys[key] &&  m_previous_keys[key];
}

int input_get_mouse_button(int button)
{
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

void input_get_mouse_position(double* x, double* y)
{
    glfwGetCursorPos(m_window, x, y);
}

// ------------------------------------------------------------------
// tempo
// ------------------------------------------------------------------

static double last_time = 0.0;
static float  delta     = 0.0f;

void time_init()
{
    last_time = glfwGetTime();
    delta = 0.0f;
}

float time_total()
{
    return (float)glfwGetTime();
}

float time_delta()
{
    return delta;
}

void time_update()
{
    double now = glfwGetTime();
    delta      = (float)(now - last_time);
    last_time  = now;
}



// ------------------------------------------------------------------
// shader
// ------------------------------------------------------------------

typedef struct
{
    unsigned int id;
} shader_t;

static GLint uniform_loc(GLuint prog, const char* name)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc == -1)
        printf("Warning: uniform '%s' not found.\n", name);
    return loc;
}

static void set_uniform_mat4(GLuint prog, const char* name, mat4* mat)
{
    glUniformMatrix4fv(uniform_loc(prog, name), 1, GL_FALSE, mat->m);
}

static void set_uniform_vec3(GLuint prog, const char* name, vec3* v)
{
    glUniform3f(uniform_loc(prog, name), v->x, v->y, v->z);
}

static void set_uniform_int(GLuint prog, const char* name, int v)
{
    glUniform1i(uniform_loc(prog, name), v);
}

static shader_t compile_shader(const char* vs, const char* fs)
{
    shader_t shader = {0};
    char info_log[512];
    int  success;

    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vs, NULL);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(vertex, 512, NULL, info_log);
        printf("ERROR::VERTEX\n%s\n", info_log);
    }

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fs, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(fragment, 512, NULL, info_log);
        printf("ERROR::FRAGMENT\n%s\n", info_log);
    }

    shader.id = glCreateProgram();
    glAttachShader(shader.id, vertex);
    glAttachShader(shader.id, fragment);
    glLinkProgram(shader.id);
    glGetProgramiv(shader.id, GL_LINK_STATUS, &success);

    if (!success)
    {
        glGetProgramInfoLog(shader.id, 512, NULL, info_log);
        printf("ERROR::LINKING\n%s\n", info_log);
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return shader;
}

shader_t shader_create_from_src(const char* vs, const char* fs)
{
    return compile_shader(vs, fs);
}

// ------------------------------------------------------------------
// vertex / mesh
// ------------------------------------------------------------------

typedef struct
{
    vec3  position;
    vec4  color;
    vec2  tex_coord;
    vec3  normal;
    float tex_index;
} vertex3d_t;

typedef struct
{
    GLuint   vao, vbo, ibo;
    uint32_t index_count;
} mesh_t;

static void setup_vertex_attribs()
{
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, color));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, tex_coord));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, tex_index));
    glEnableVertexAttribArray(4);
}

mesh_t mesh_create(vertex3d_t* vertices, uint32_t vertex_count,
                   uint32_t*   indices,  uint32_t index_count)
{
    mesh_t mesh = {0};
    mesh.index_count = index_count;

    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);

    glGenBuffers(1, &mesh.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(vertex3d_t), vertices, GL_STATIC_DRAW);
    setup_vertex_attribs();

    glGenBuffers(1, &mesh.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_count * sizeof(uint32_t), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    return mesh;
}

void mesh_destroy(mesh_t* mesh)
{
    glDeleteBuffers(1, &mesh->vbo);
    glDeleteBuffers(1, &mesh->ibo);
    glDeleteVertexArrays(1, &mesh->vao);
}

void mesh_draw(mesh_t* mesh)
{
    glBindVertexArray(mesh->vao);
    glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

// ------------------------------------------------------------------
// texturas
// ------------------------------------------------------------------

typedef struct
{
    GLuint id;
    int width, height;
} texture_t;

texture_t texture_load(const char* path)
{
    texture_t tex = {0};
    stbi_set_flip_vertically_on_load(1);
    int channels;
    unsigned char* data = stbi_load(path, &tex.width, &tex.height, &channels, 4);

    if (!data)
    {
        printf("Erro ao carregar textura: %s\n", path);
        return tex;
    }

    glGenTextures(1, &tex.id);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);

    glBindTexture(GL_TEXTURE_2D, 0);

    printf("Textura carregada: %s (%dx%d)\n", path, tex.width, tex.height);
    return tex;
}

void texture_destroy(texture_t* tex)
{
    glDeleteTextures(1, &tex->id);
    tex->id = 0;
}

void texture_bind(texture_t* tex, int slot)
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, tex->id);
}

// ------------------------------------------------------------------
// renderer3d
// ------------------------------------------------------------------

#define MAX_QUADS    1000
#define MAX_VERTICES (MAX_QUADS * 4)
#define MAX_INDICES  (MAX_QUADS * 6)
#define MAX_TEXTURES 16

typedef struct
{
    GLuint vao, vbo, ibo;

    uint32_t    index_count;
    vertex3d_t* vertex_buffer_base;
    vertex3d_t* vertex_buffer_ptr;

    GLuint   texture_slots[MAX_TEXTURES];
    uint32_t texture_slot_index;

    vec4 quad_vertex_positions[4];

    GLuint current_shader;
    mat4   current_view;
    mat4   current_projection;
} renderer3d_t;

void renderer3d_init(renderer3d_t* r)
{
    memset(r, 0, sizeof(renderer3d_t));
    r->texture_slot_index = 1;

    r->quad_vertex_positions[0] = (vec4){ -0.5f, -0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[1] = (vec4){ +0.5f, -0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[2] = (vec4){ +0.5f, +0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[3] = (vec4){ -0.5f, +0.5f, 0.0f, 1.0f };

    glGenVertexArrays(1, &r->vao);
    glBindVertexArray(r->vao);

    glGenBuffers(1, &r->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_VERTICES * sizeof(vertex3d_t), NULL, GL_DYNAMIC_DRAW);
    setup_vertex_attribs();

    uint32_t* indices = malloc(MAX_INDICES * sizeof(uint32_t));
    uint32_t  offset  = 0;
    for (uint32_t i = 0; i < MAX_INDICES; i += 6)
    {
        indices[i+0] = offset + 0; indices[i+1] = offset + 1; indices[i+2] = offset + 2;
        indices[i+3] = offset + 2; indices[i+4] = offset + 3; indices[i+5] = offset + 0;
        offset += 4;
    }

    glGenBuffers(1, &r->ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, MAX_INDICES * sizeof(uint32_t), indices, GL_STATIC_DRAW);
    free(indices);

    r->vertex_buffer_base = malloc(MAX_VERTICES * sizeof(vertex3d_t));
    glBindVertexArray(0);
}

void renderer3d_destroy(renderer3d_t* r)
{
    free(r->vertex_buffer_base);
    glDeleteBuffers(1, &r->vbo);
    glDeleteBuffers(1, &r->ibo);
    glDeleteVertexArrays(1, &r->vao);
}

void renderer3d_flush(renderer3d_t* r)
{
    glUseProgram(r->current_shader);

    for (uint32_t i = 0; i < r->texture_slot_index; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, r->texture_slots[i]);

        char name[32];
        snprintf(name, sizeof(name), "u_textures[%u]", i);
        glUniform1i(glGetUniformLocation(r->current_shader, name), (int)i);
    }

    glBindVertexArray(r->vao);
    glDrawElements(GL_TRIANGLES, r->index_count, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

void renderer3d_begin_batch(renderer3d_t* r, GLuint shader_id,
                             mat4* view, mat4* projection)
{
    r->index_count        = 0;
    r->vertex_buffer_ptr  = r->vertex_buffer_base;
    r->texture_slot_index = 1;   // slot 0 reservado para "sem textura"

    r->current_shader     = shader_id;
    r->current_view       = *view;
    r->current_projection = *projection;

    vec3 blend_color = vec3_from_scalar(1.0);

    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_view",       view);
    set_uniform_mat4(shader_id, "u_projection", projection);
    set_uniform_vec3(shader_id, "u_light_dir", &MAIN_LIGHT_DIR);
    set_uniform_vec3(shader_id, "u_light_color", &MAIN_LIGHT_COLOR);
    set_uniform_vec3(shader_id, "u_blend_color", &blend_color);

    mat4 identity = mat4_identity();
    set_uniform_mat4(shader_id, "u_model", &identity);
    // sem u_use_texture aqui
}

void renderer3d_end_batch(renderer3d_t* r)
{
    ptrdiff_t size = (uint8_t*)r->vertex_buffer_ptr - (uint8_t*)r->vertex_buffer_base;
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, r->vertex_buffer_base);
    renderer3d_flush(r);
}

void renderer3d_draw_quad(renderer3d_t* r,
                          vec3 position,
                          vec3 rotation,
                          vec2 size,
                          vec3 normal_local,
                          GLuint texture_id,      // 0 = sem textura
                          vec4 color,
                          texture_t* texture)     // pode ser NULL
{
    if (r->index_count >= MAX_INDICES)
    {
        renderer3d_end_batch(r);
        renderer3d_begin_batch(r, r->current_shader,
                               &r->current_view, &r->current_projection);
    }

    // --- registra a textura no slot ---
    float tex_index = 0.0f;

    if (texture && texture->id != 0)
    {
        for (uint32_t i = 1; i < r->texture_slot_index; i++)
        {
            if (r->texture_slots[i] == texture->id)
            {
                tex_index = (float)i;
                break;
            }
        }

        if (tex_index == 0.0f)
        {
            if (r->texture_slot_index >= MAX_TEXTURES)
            {
                renderer3d_end_batch(r);
                renderer3d_begin_batch(r, r->current_shader,
                                       &r->current_view, &r->current_projection);
            }
            tex_index = (float)r->texture_slot_index;
            r->texture_slots[r->texture_slot_index++] = texture->id;
        }
    }

    // --- transform + normal ---
    float transform[16];
    vec3  world_normal;
    build_transform_with_normal(transform, position, rotation,
                                size.x, size.y, normal_local, &world_normal);

    vec2 tex_coords[4] = {
        { 0.0f, 0.0f }, { 1.0f, 0.0f },
        { 1.0f, 1.0f }, { 0.0f, 1.0f }
    };

    for (int i = 0; i < 4; i++)
    {
        r->vertex_buffer_ptr->position  = transform_vec4(transform,
                                              r->quad_vertex_positions[i]);
        r->vertex_buffer_ptr->color     = color;
        r->vertex_buffer_ptr->tex_coord = tex_coords[i];
        r->vertex_buffer_ptr->normal    = world_normal;
        r->vertex_buffer_ptr->tex_index = tex_index;  // 0 = cor pura
        r->vertex_buffer_ptr++;
    }

    r->index_count += 6;
}

void renderer3d_draw_mesh(mesh_t* mesh, GLuint shader_id, mat4* model,
                           mat4* view, mat4* projection,
                           texture_t* texture, vec3* blend_color)
{
    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_model",       model);
    set_uniform_mat4(shader_id, "u_view",        view);
    set_uniform_mat4(shader_id, "u_projection",  projection);
    set_uniform_vec3(shader_id, "u_light_dir", &MAIN_LIGHT_DIR);
    set_uniform_vec3(shader_id, "u_light_color",&MAIN_LIGHT_COLOR);
    set_uniform_vec3(shader_id, "u_blend_color", blend_color);

    if (texture && texture->id != 0)
    {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture->id);
        glUniform1i(glGetUniformLocation(shader_id, "u_textures[1]"), 1);
    }
    else
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUniform1i(glGetUniformLocation(shader_id, "u_textures[0]"), 0);
    }

    mesh_draw(mesh);
}

// ------------------------------------------------------------------
// game objects
// ------------------------------------------------------------------

typedef struct
{
    vec3 position, rotation, scale;
} transform_t;

transform_t transform_identity()
{
    transform_t t;
    t.position = vec3_zero();
    t.rotation = vec3_zero();
    t.scale    = vec3_one();
    return t;
}

typedef struct
{
    transform_t transform;
    vec3        speed;
    vec3        angular_speed;
    mesh_t*     mesh;
    texture_t*  texture;
    vec3        blend_color;
    int         tag;           // OBJECT_TAG_*
} game_object_t;

game_object_t game_object_create()
{
    game_object_t o;
    o.transform     = transform_identity();
    o.speed         = vec3_zero();
    o.angular_speed = vec3_zero();
    o.mesh          = NULL;
    o.texture       = NULL;
    o.blend_color   = vec3_from_scalar(1.0);
    return o;
}

void game_object_update(game_object_t* o, float dt)
{
    vec3 delta = vec3_multiply_scalar(&o->speed, dt);
    vec3 delta_angle = vec3_multiply_scalar(&o->angular_speed, dt);
    o->transform.position = vec3_add(&o->transform.position, &delta);
    o->transform.rotation = vec3_add(&o->transform.rotation, &delta_angle);
}

// ------------------------------------------------------------------
// .obj parser
// ------------------------------------------------------------------

typedef struct
{
    vec3* positions;
    vec2* texcoords;
    vec3* normals;
    int position_count, texcoord_count, normal_count;
    int* pos_indices;
    int* tex_indices;
    int* nor_indices;
    int  face_count;
} obj_data_t;

void load_obj(const char* path, obj_data_t* obj)
{
    FILE* fptr = fopen(path, "r");

    if (!fptr)
    {
        printf("Error: could not open %s\n", path);
        return;
    }

    int cap_v = 65536, cap_vt = 65536, cap_vn = 65536, cap_f = 131072;
    obj->positions   = malloc(cap_v  * sizeof(vec3));
    obj->texcoords   = malloc(cap_vt * sizeof(vec2));
    obj->normals     = malloc(cap_vn * sizeof(vec3));
    obj->pos_indices = malloc(cap_f * 3 * sizeof(int));
    obj->tex_indices = malloc(cap_f * 3 * sizeof(int));
    obj->nor_indices = malloc(cap_f * 3 * sizeof(int));
    obj->position_count = obj->texcoord_count = obj->normal_count = obj->face_count = 0;

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), fptr))
    {
        if (strncmp(buffer, "vn ", 3) == 0)
        {
            vec3 n;
            if (sscanf(buffer + 3, "%f %f %f", &n.x, &n.y, &n.z) == 3)
                obj->normals[obj->normal_count++] = n;
        }
        else if (strncmp(buffer, "vt ", 3) == 0)
        {
            vec2 t;
            if (sscanf(buffer + 3, "%f %f", &t.x, &t.y) == 2)
                obj->texcoords[obj->texcoord_count++] = t;
        }
        else if (strncmp(buffer, "v ", 2) == 0)
        {
            vec3 v;
            if (sscanf(buffer + 2, "%f %f %f", &v.x, &v.y, &v.z) == 3)
                obj->positions[obj->position_count++] = v;
        }
        else if (strncmp(buffer, "f ", 2) == 0)
        {
            int pi[3], ti[3], ni[3];
            int r = sscanf(buffer + 2,
                "%d/%d/%d %d/%d/%d %d/%d/%d",
                &pi[0],&ti[0],&ni[0], &pi[1],&ti[1],&ni[1], &pi[2],&ti[2],&ni[2]);
            if (r != 9)
            {
                r = sscanf(buffer + 2,
                    "%d//%d %d//%d %d//%d",
                    &pi[0],&ni[0], &pi[1],&ni[1], &pi[2],&ni[2]);
                ti[0] = ti[1] = ti[2] = 1;
                if (r != 6) continue;
            }
            int base = obj->face_count * 3;
            for (int i = 0; i < 3; i++)
            {
                obj->pos_indices[base+i] = pi[i] - 1;
                obj->tex_indices[base+i] = ti[i] - 1;
                obj->nor_indices[base+i] = ni[i] - 1;
            }
            obj->face_count++;
        }
    }
    fclose(fptr);
    printf("load_obj: %d verts, %d texcoords, %d normals, %d faces\n",
           obj->position_count, obj->texcoord_count, obj->normal_count, obj->face_count);
}

void obj_data_free(obj_data_t* obj)
{
    free(obj->positions);
    free(obj->texcoords);
    free(obj->normals);
    free(obj->pos_indices);
    free(obj->tex_indices);
    free(obj->nor_indices);
    memset(obj, 0, sizeof(obj_data_t));
}

// ------------------------------------------------------------------
// asset loading
// ------------------------------------------------------------------

typedef struct
{
    mesh_t    mesh;
    texture_t texture;
    bool      has_texture;
} model_asset_t;

model_asset_t model_asset_load(const char* obj_path, const char* texture_path, float tex_index)
{
    model_asset_t asset = {0};

    obj_data_t data = {0};
    load_obj(obj_path, &data);

    if (data.face_count == 0)
    {
        printf("model_asset_load: falha ao carregar %s\n", obj_path);
        obj_data_free(&data);
        return asset;
    }

    int total_verts = data.face_count * 3;
    vertex3d_t* verts = malloc(total_verts * sizeof(vertex3d_t));
    uint32_t*   idx   = malloc(total_verts * sizeof(uint32_t));

    for (int i = 0; i < total_verts; i++)
    {
        verts[i].position  = data.positions[data.pos_indices[i]];
        verts[i].tex_coord = data.texcoords[data.tex_indices[i]];
        verts[i].normal    = data.normals  [data.nor_indices[i]];
        verts[i].color     = (vec4){ 1.0f, 1.0f, 1.0f, 1.0f };
        verts[i].tex_index = tex_index;
        idx[i]             = (uint32_t)i;
    }

    asset.mesh = mesh_create(verts, total_verts, idx, total_verts);
    free(verts);
    free(idx);
    obj_data_free(&data);

    if (texture_path != NULL)
    {
        asset.texture     = texture_load(texture_path);
        asset.has_texture = (asset.texture.id != 0);
    }

    return asset;
}

void model_asset_destroy(model_asset_t* asset)
{
    mesh_destroy(&asset->mesh);
    if (asset->has_texture)
        texture_destroy(&asset->texture);
    memset(asset, 0, sizeof(model_asset_t));
}

game_object_t model_asset_instantiate(model_asset_t* asset)
{
    game_object_t obj = game_object_create();
    obj.mesh    = &asset->mesh;
    obj.texture = asset->has_texture ? &asset->texture : NULL;
    return obj;
}

// ------------------------------------------------------------------
// vec3 / math
// ------------------------------------------------------------------

vec3 vec3_zero()
{
    return (vec3){ 0, 0, 0 };
}

vec3 vec3_one()
{
    return (vec3){ 1, 1, 1 };
}

vec3 vec3_add(vec3* a, vec3* b)
{
    return (vec3)
    {
        a->x+b->x,
        a->y+b->y,
        a->z+b->z
    };
}

vec3 vec3_subtract(vec3* a, vec3* b)
{
    return (vec3)
    {
        a->x-b->x,
        a->y-b->y,
        a->z-b->z
    };
}

vec3 vec3_multiply_scalar(vec3* v, float s)
{
    return (vec3)
    {
        v->x*s,
        v->y*s,
        v->z*s
    };
}

vec3 vec3_cross(vec3* a, vec3* b)
{
    return (vec3)
    {
        a->y*b->z - a->z*b->y,
        a->z*b->x - a->x*b->z,
        a->x*b->y - a->y*b->x
    };
}

float vec3_dot(vec3* a, vec3* b)
{
    return a->x*b->x + a->y*b->y + a->z*b->z;
}

float lerp(float a, float b, float t)
{
    return (1-t)*a + t*b;
}

vec3 vec3_lerp(vec3* a, vec3* b, float t)
{
    return (vec3)
    {
        lerp(a->x,b->x,t),
        lerp(a->y,b->y,t),
        lerp(a->z,b->z,t)
    };
}

vec3 vec3_from_scalar(float scalar)
{
    return (vec3){ scalar, scalar, scalar };
}

vec3 vec3_negate(vec3* vec)
{
    return (vec3)
    {
        -vec->x,
        -vec->y,
        -vec->z
    };
}

float vec3_length(vec3* vec)
{
    return sqrt(vec->x * vec->x + vec->y * vec->y + vec->z * vec->z);
}

vec3 vec3_normalize(vec3* vec)
{
    float length = vec3_length(vec);
    return (vec3)
    {
        vec->x / length,
        vec->y / length,
        vec->z / length,
    };
}

vec3 vec3_stretch_along(vec3* a, vec3* b, float u)
{
    float dot_ab = vec3_dot(a, b);
    float dot_bb = vec3_dot(b, b);
    float scale = u * dot_ab / dot_bb;
    return vec3_multiply_scalar(b, scale);
}

// ------------------------------------------------------------------
// mat4
// ------------------------------------------------------------------

mat4 mat4_identity()
{
    return (mat4)
    {
        .m =
        { 1,0,0,0,
          0,1,0,0,
          0,0,1,0,
          0,0,0,1
        }
    };
}

mat4 mat4_multiply(mat4* a, mat4* b)
{
    mat4 result = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                result.m[j*4+i] += a->m[k*4+i] * b->m[j*4+k];
    return result;
}

mat4 mat4_translate(mat4* mat, vec3* vec)
{
    mat4 result = *mat;
    result.m[12] += vec->x;
    result.m[13] += vec->y;
    result.m[14] += vec->z;
    return result;
}

mat4 mat4_rotate_x(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[5]  =  cosf(angle);
    rot.m[9]  = -sinf(angle);
    rot.m[6]  =  sinf(angle);
    rot.m[10] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate_y(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[0]  =  cosf(angle);
    rot.m[8]  =  sinf(angle);
    rot.m[2]  = -sinf(angle);
    rot.m[10] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate_z(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[0] =  cosf(angle);
    rot.m[4] = -sinf(angle);
    rot.m[1] =  sinf(angle);
    rot.m[5] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate(mat4* mat, vec3* axis, float angle)
{
    vec3 n = vec3_normalize(axis);
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;

    mat4 rot = mat4_identity();
    rot.m[0]  = t * n.x * n.x + c;
    rot.m[1]  = t * n.x * n.y + s * n.z;
    rot.m[2]  = t * n.x * n.z - s * n.y;
    rot.m[4]  = t * n.x * n.y - s * n.z;
    rot.m[5]  = t * n.y * n.y + c;
    rot.m[6]  = t * n.y * n.z + s * n.x;
    rot.m[8]  = t * n.x * n.z + s * n.y;
    rot.m[9]  = t * n.y * n.z - s * n.x;
    rot.m[10] = t * n.z * n.z + c;

    return mat4_multiply(mat, &rot);
}

mat4 mat4_scale(mat4* mat, vec3* vec)
{
    mat4 s = mat4_identity();
    s.m[0]  = vec->x;
    s.m[5]  = vec->y;
    s.m[10] = vec->z;
    return mat4_multiply(mat, &s);
}

mat4 mat4_perspective(float fov, float aspect, float near, float far)
{
    mat4 result;
    memset(result.m, 0, sizeof(result.m));
    float f = 1.0f / tanf(fov * 0.5f);
    result.m[0]  = f / aspect;
    result.m[5]  = f;
    result.m[10] = -(far + near) / (far - near);
    result.m[11] = -1.0f;
    result.m[14] = -(2.0f * far * near) / (far - near);
    return result;
}

mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up)
{
    mat4 result;
    memset(result.m, 0, sizeof(result.m));

    vec3 f =
    {
        center.x-eye.x,
        center.y-eye.y,
        center.z-eye.z
    };

    float fl = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
    f.x /= fl; f.y /= fl; f.z /= fl;

    vec3 s =
    {
        f.y*up.z-f.z*up.y,
        f.z*up.x-f.x*up.z,
        f.x*up.y-f.y*up.x
    };
    float sl = sqrtf(s.x*s.x + s.y*s.y + s.z*s.z);
    s.x /= sl; s.y /= sl; s.z /= sl;

    vec3 u = { s.y*f.z-s.z*f.y, s.z*f.x-s.x*f.z, s.x*f.y-s.y*f.x };

    result.m[0]  =  s.x;
    result.m[4]  =  s.y;
    result.m[8]  =  s.z;
    result.m[1]  =  u.x;
    result.m[5]  =  u.y;
    result.m[9]  =  u.z;
    result.m[2]  = -f.x;
    result.m[6]  = -f.y;
    result.m[10] = -f.z;
    result.m[12] = -(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    result.m[13] = -(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    result.m[14] =  (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    result.m[15] =  1.0f;

    return result;
}

// ------------------------------------------------------------------
// camera
// ------------------------------------------------------------------

const vec3 CAMERA_UP = (vec3){ 0, 1, 0 };
bool camera_free = false;

typedef struct
{
    vec3  position;
    vec3  last_position;
    float yaw, last_yaw;
    float pitch, last_pitch;
    float speed, sensitivity;
} camera;

vec3 camera_get_forward(camera* cam)
{
    vec3 dir =
    {
        cosf(cam->yaw) * cosf(cam->pitch),
        sinf(cam->pitch),
        sinf(cam->yaw) * cosf(cam->pitch)
    };
    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;
    return dir;
}

vec3 camera_get_right(camera* cam)
{
    vec3 fwd   = camera_get_forward(cam);
    vec3 right = vec3_cross(&fwd, (vec3*)&CAMERA_UP);
    float len  = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= len; right.y /= len; right.z /= len;
    return right;
}

vec2 vec2_zero()
{
    return (vec2){ 0, 0 };
}

vec2 vec2_one()
{
    return (vec2){ 1, 1 };
}

vec2 vec2_from_scalar(float s)
{
    return (vec2){ s, s };
}

vec2 vec2_add(vec2* a, vec2* b)
{
    return (vec2){ a->x + b->x, a->y + b->y };
}

vec2 vec2_subtract(vec2* a, vec2* b)
{
    return (vec2){ a->x - b->x, a->y - b->y };
}

vec2 vec2_multiply_scalar(vec2* v, float s)
{
    return (vec2){ v->x * s, v->y * s };
}

vec2 vec2_multiply(vec2* a, vec2* b)
{
    return (vec2){ a->x * b->x, a->y * b->y };
}

vec2 vec2_divide_scalar(vec2* v, float s)
{
    return (vec2){ v->x / s, v->y / s };
}

vec2 vec2_negate(vec2* v)
{
    return (vec2){ -v->x, -v->y };
}

float vec2_dot(vec2* a, vec2* b)
{
    return a->x * b->x + a->y * b->y;
}

float vec2_length_sq(vec2* v)
{
    return v->x * v->x + v->y * v->y;
}

float vec2_length(vec2* v)
{
    return sqrtf(vec2_length_sq(v));
}

vec2 vec2_normalize(vec2* v)
{
    float len = vec2_length(v);
    if (len < 0.00001f) return vec2_zero();
    return (vec2){ v->x / len, v->y / len };
}

vec2 vec2_lerp(vec2* a, vec2* b, float t)
{
    return (vec2){
        a->x + (b->x - a->x) * t,
        a->y + (b->y - a->y) * t,
    };
}

// ------------------------------------------------------------------
// game object array (dinâmico)
// ------------------------------------------------------------------

typedef struct
{
    game_object_t* data;
    int            capacity;
    int            size;
} game_object_array_t;

void game_object_array_init(game_object_array_t* arr, int initial_capacity)
{
    arr->data     = malloc(initial_capacity * sizeof(game_object_t));
    arr->capacity = initial_capacity;
    arr->size     = 0;
}

void game_object_array_free(game_object_array_t* arr)
{
    free(arr->data);
    arr->data     = NULL;
    arr->capacity = 0;
    arr->size     = 0;
}

int game_object_array_push(game_object_array_t* arr, game_object_t obj)
{
    if (arr->size >= arr->capacity)
    {
        int new_capacity = arr->capacity * 2;
        if (new_capacity == 0) new_capacity = 16;
        game_object_t* new_data = realloc(arr->data, new_capacity * sizeof(game_object_t));
        if (!new_data)
        {
            printf("game_object_array: falha ao realocar\n");
            return -1;
        }
        arr->data     = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size] = obj;
    return arr->size++;
}

game_object_t* game_object_array_get(game_object_array_t* arr, int index)
{
    if (index < 0 || index >= arr->size) return NULL;
    return &arr->data[index];
}

void game_object_array_remove(game_object_array_t* arr, int index)
{
    if (index < 0 || index >= arr->size) return;

    // Move o último elemento para a posição removida
    arr->data[index] = arr->data[arr->size - 1];
    arr->size--;
}

void game_object_array_clear(game_object_array_t* arr)
{
    arr->size = 0;
}

// ------------------------------------------------------------------
// vec3 array (dinâmico)
// ------------------------------------------------------------------

typedef struct
{
    vec3* data;
    int   capacity;
    int   size;
} vec3_array_t;

void vec3_array_init(vec3_array_t* arr, int initial_capacity)
{
    arr->data     = malloc(initial_capacity * sizeof(vec3));
    arr->capacity = initial_capacity;
    arr->size     = 0;
}

void vec3_array_free(vec3_array_t* arr)
{
    free(arr->data);
    arr->data     = NULL;
    arr->capacity = 0;
    arr->size     = 0;
}

int vec3_array_push(vec3_array_t* arr, vec3 v)
{
    if (arr->size >= arr->capacity)
    {
        int new_capacity = arr->capacity * 2;
        if (new_capacity == 0) new_capacity = 16;
        vec3* new_data = realloc(arr->data, new_capacity * sizeof(vec3));
        if (!new_data)
        {
            printf("vec3_array: falha ao realocar\n");
            return -1;
        }
        arr->data     = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size] = v;
    return arr->size++;
}

vec3* vec3_array_get(vec3_array_t* arr, int index)
{
    if (index < 0 || index >= arr->size) return NULL;
    return &arr->data[index];
}

void vec3_array_remove(vec3_array_t* arr, int index)
{
    if (index < 0 || index >= arr->size) return;

    // Move o último elemento para a posição removida
    arr->data[index] = arr->data[arr->size - 1];
    arr->size--;
}

void vec3_array_clear(vec3_array_t* arr)
{
    arr->size = 0;
}


// ------------------------------------------------------------------
// game world
// ------------------------------------------------------------------

typedef struct
{
    game_object_array_t objects;
} game_world_t;

void game_world_init(game_world_t* world)
{
    game_object_array_init(&world->objects, 1024);
}

int game_world_add(game_world_t* world, game_object_t object)
{
    return game_object_array_push(&world->objects, object);
}

void game_world_remove(game_world_t* world, int index)
{
    game_object_array_remove(&world->objects, index);
}

const float ROOM_X_BOUND = 500.0;
void game_world_update(game_world_t* world, float dt)
{
    for (int i = 0; i < world->objects.size; i++)
    {
        game_object_t* obj = &world->objects.data[i];
        game_object_update(obj, dt);
        if (obj->transform.position.z < -20.0f ||
            obj->transform.position.z > ROOM_X_BOUND)
        {
            game_object_array_remove(&world->objects, i);
            i--;
        }
    }
}

void game_world_render(game_world_t* world, GLuint shader, mat4* view, mat4* projection)
{
    // Primeiro renderiza objetos opacos
    for (int i = 0; i < world->objects.size; i++)
    {
        game_object_t* obj = &world->objects.data[i];
        if (!obj->mesh) continue;

        // Verifica se o objeto é transparente
        // Os cubos têm alpha = 0.5 nos vértices, então são transparentes
        bool is_transparent = true;

        if (is_transparent) continue; // Pula transparentes por agora

        // S
        mat4 model = mat4_identity();
        model.m[0]  = obj->transform.scale.x;
        model.m[5]  = obj->transform.scale.y;
        model.m[10] = obj->transform.scale.z;

        // R (Z * Y * X aplicado à esquerda)
        model = mat4_rotate_x(&model, obj->transform.rotation.x);
        model = mat4_rotate_y(&model, obj->transform.rotation.y);
        model = mat4_rotate_z(&model, obj->transform.rotation.z);

        // T
        model.m[12] = obj->transform.position.x;
        model.m[13] = obj->transform.position.y;
        model.m[14] = obj->transform.position.z;

        renderer3d_draw_mesh(obj->mesh, shader, &model, view, projection, obj->texture, &obj->blend_color);
    }

    // Agora renderiza objetos transparentes com depth writing desabilitado
    glDepthMask(GL_FALSE);
    for (int i = 0; i < world->objects.size; i++)
    {
        game_object_t* obj = &world->objects.data[i];
        if (!obj->mesh) continue;

        // Verifica se o objeto é transparente
        bool is_transparent = true;

        if (!is_transparent) continue; // Pula opacos

        // S
        mat4 model = mat4_identity();
        model.m[0]  = obj->transform.scale.x;
        model.m[5]  = obj->transform.scale.y;
        model.m[10] = obj->transform.scale.z;

        // R (Z * Y * X aplicado à esquerda)
        model = mat4_rotate_x(&model, obj->transform.rotation.x);
        model = mat4_rotate_y(&model, obj->transform.rotation.y);
        model = mat4_rotate_z(&model, obj->transform.rotation.z);

        // T
        model.m[12] = obj->transform.position.x;
        model.m[13] = obj->transform.position.y;
        model.m[14] = obj->transform.position.z;

        renderer3d_draw_mesh(obj->mesh, shader, &model, view, projection, obj->texture, &obj->blend_color);
    }
    glDepthMask(GL_TRUE);
}

game_object_t* game_world_get_object(game_world_t* world, int object_id)
{
    return game_object_array_get(&world->objects, object_id);
}

// ------------------------------------------------------------------
// random
// ------------------------------------------------------------------

void rand_init()
{
    srand((unsigned int)time(NULL));
}

void rand_init_seed(unsigned int seed)
{
    srand(seed);
}

int rand_int(int min, int max)
{
    return min + rand() % (max - min + 1);
}

float rand_float01()
{
    return (float)rand() / (float)RAND_MAX;
}

float rand_float(float min, float max)
{
    return min + rand_float01() * (max - min);
}

int rand_chance(float p)
{
    return rand_float01() < p;
}

vec3 rand_vec3(float min, float max)
{
    return (vec3){
        rand_float(min, max),
        rand_float(min, max),
        rand_float(min, max)
    };
}

int sign(int n)
{
    if (n > 0) return 1;
    else if (n < 0) return -1;
    return n;
}

// ------------------------------------------------------------------
// snake
// ------------------------------------------------------------------

typedef struct
{
    vec3_array_t segments;      // posições dos segmentos (índice 0 = cabeça)
    vec3          head_forward; // direção de movimento
    vec3          head_right;   // direção lateral (para rotação nas bordas)
    vec3          head_up;      // direção para cima
    vec3          current_face_normal; // normal da face atual
    float         last_tick;   // tempo do último tick
    float         tick_interval; // intervalo entre ticks (0.5s)
    float         cube_size;   // tamanho do cubo onde a cobra se move
    float         step_size;   // tamanho de cada passo
    bool          paused;
} snake_t;

void snake_init(snake_t* snake, float cube_size, float tick_interval)
{
    vec3_array_init(&snake->segments, 64);
    snake->head_forward = (vec3){ 0.0f, 1.0f, 0.0f };
    snake->head_right   = (vec3){ 1.0f, 0.0f, 0.0f };
    snake->head_up      = (vec3){ 0.0f, 1.0f, 0.0f };
    snake->current_face_normal = (vec3){ 0.0f, 0.0f, 1.0f }; // Começa na face frontal
    snake->last_tick    = 0.0f;
    snake->tick_interval = tick_interval;
    snake->cube_size    = cube_size;
    snake->step_size    = 1.1f;
    snake->paused = false;

    // Começa no centro da face frontal
    vec3_array_push(&snake->segments, (vec3){ 0.0f, 0.0f, cube_size / 2.0f });
}

void snake_free(snake_t* snake)
{
    vec3_array_free(&snake->segments);
}

// Rotaciona um vetor em torno de um eixo por um ângulo
vec3 vec3_rotate_around_axis(vec3 v, vec3 axis, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);
    vec3  n = vec3_normalize(&axis);

    // Fórmula de rotação de Rodrigues
    vec3 term1 = vec3_multiply_scalar(&v, c);
    vec3 cross = vec3_cross(&n, &v);
    vec3 term2 = vec3_multiply_scalar(&cross, s);
    vec3 dot_vec = vec3_multiply_scalar(&n, vec3_dot(&n, &v));
    vec3 term3 = vec3_multiply_scalar(&dot_vec, 1.0f - c);

    vec3 result = vec3_add(&term1, &term2);
    result = vec3_add(&result, &term3);
    return result;
}

void snake_update(snake_t* snake, float current_time)
{
    if (snake->paused)
    {
        return;
    }

    vec3 forward = snake->head_forward;
    vec3 normal  = snake->current_face_normal;
    vec3 right   = vec3_cross(&normal, &forward);
    right = vec3_normalize(&right);

    vec3 new_forward = forward;

    if (input_get_key_down(GLFW_KEY_A))
        new_forward = vec3_rotate_around_axis(new_forward, normal,  PI/2.0f);
    if (input_get_key_down(GLFW_KEY_D))
        new_forward = vec3_rotate_around_axis(new_forward, normal, -PI/2.0f);
    if (input_get_key_down(GLFW_KEY_W))
        new_forward = vec3_rotate_around_axis(new_forward, right,   PI/2.0f);
    if (input_get_key_down(GLFW_KEY_S))
        new_forward = vec3_rotate_around_axis(new_forward, right,  -PI/2.0f);

    new_forward = vec3_normalize(&new_forward);

    float dot_fwd = vec3_dot(&forward, &new_forward);
    if (dot_fwd < -0.9f) {
        new_forward = forward;
    }

    float dot_norm = vec3_dot(&normal, &new_forward);
    if (fabsf(dot_norm) > 0.01f) {
        new_forward = forward;
    }

    snake->head_forward = new_forward;

    if (current_time - snake->last_tick < snake->tick_interval)
        return;
    snake->last_tick = current_time;

    vec3* head = &snake->segments.data[0];
    vec3 delta = vec3_multiply_scalar(&snake->head_forward, snake->step_size);
    vec3 new_head = vec3_add(head, &delta);

    float half = snake->cube_size * 0.5f + 1;
    int   hit_axis = -1;
    int   hit_sign = 0;

    if      (fabsf(new_head.x) > half) { hit_axis = 0; hit_sign = (new_head.x > 0) ? 1 : -1; }
    else if (fabsf(new_head.y) > half) { hit_axis = 1; hit_sign = (new_head.y > 0) ? 1 : -1; }
    else if (fabsf(new_head.z) > half) { hit_axis = 2; hit_sign = (new_head.z > 0) ? 1 : -1; }

    if (hit_axis != -1)
    {
        new_head = *head;

        vec3 edge_axis;
        vec3 new_normal;
        float angle;

        vec3 old_normal = snake->current_face_normal;

        if (hit_axis == 0)
        {
            new_normal = (vec3){ (float)hit_sign, 0.0f, 0.0f };
            edge_axis = vec3_cross(&old_normal, &new_normal);
            edge_axis = vec3_normalize(&edge_axis);
        }
        else if (hit_axis == 1)
        {
            new_normal = (vec3){ 0.0f, (float)hit_sign, 0.0f };
            edge_axis = vec3_cross(&old_normal, &new_normal);
            edge_axis = vec3_normalize(&edge_axis);
        }
        else
        {
            new_normal = (vec3){ 0.0f, 0.0f, (float)hit_sign };
            edge_axis = vec3_cross(&old_normal, &new_normal);
            edge_axis = vec3_normalize(&edge_axis);
        }

        angle = PI / 2.0f;

        snake->head_forward = vec3_rotate_around_axis(snake->head_forward, edge_axis, angle);
        snake->current_face_normal = vec3_rotate_around_axis(snake->current_face_normal, edge_axis, angle);

        snake->head_forward = vec3_normalize(&snake->head_forward);
        snake->current_face_normal = vec3_normalize(&snake->current_face_normal);

        vec3 new_right = vec3_cross(&snake->current_face_normal, &snake->head_forward);
        snake->head_right = vec3_normalize(&new_right);
        snake->head_up = snake->current_face_normal;

        delta = vec3_multiply_scalar(&snake->head_forward, snake->step_size);
        new_head = vec3_add(head, &delta);
    }

    for (int i = snake->segments.size - 1; i > 0; i--)
    {
        snake->segments.data[i] = snake->segments.data[i-1];
    }

    snake->segments.data[0] = new_head;
}

void snake_grow(snake_t* snake)
{
    if (snake->segments.size > 0)
    {
        vec3* tail = vec3_array_get(&snake->segments, snake->segments.size - 1);
        vec3_array_push(&snake->segments, *tail);
    }
}

void snake_render(snake_t* snake, mesh_t* head_mesh, mesh_t* body_mesh, GLuint shader, texture_t* texture, mat4* view, mat4* projection)
{
    for (int i = 0; i < snake->segments.size; i++)
    {
        vec3* pos = vec3_array_get(&snake->segments, i);
        if (!pos) continue;

        mat4 model = mat4_identity();

        float scale = (i == 0) ? 0.8f : 0.6f;
        model.m[0]  = scale;
        model.m[5]  = scale;
        model.m[10] = scale;

        // Posição
        model.m[12] = pos->x;
        model.m[13] = pos->y;
        model.m[14] = pos->z;

        mesh_t* mesh = (i == 0) ? head_mesh : body_mesh;
        vec3 blend_color = (i == 0) ? (vec3){ 1.0f, 0.2f, 0.2f } : (vec3){ 0.2f, 0.8f, 0.2f };

        renderer3d_draw_mesh(mesh, shader, &model, view, projection, texture, &blend_color);
    }
}

// ------------------------------------------------------------------
// gerador de maças
// ------------------------------------------------------------------

bool apple_exists = false;

const vec3 NORMAL_VECTORS[6] = {
    (vec3){  1.0f,  0.0f,  0.0f },
    (vec3){ -1.0f,  0.0f,  0.0f },
    (vec3){  0.0f,  1.0f,  0.0f },
    (vec3){  0.0f, -1.0f,  0.0f },
    (vec3){  0.0f,  0.0f,  1.0f },
    (vec3){  0.0f,  0.0f, -1.0f },
};

const vec3 RIGHT_VECTORS[6] = {
    (vec3){  0.0f,  0.0f, -1.0f },
    (vec3){  0.0f,  0.0f,  1.0f },
    (vec3){  1.0f,  0.0f,  0.0f },
    (vec3){  1.0f,  0.0f,  0.0f },
    (vec3){  1.0f,  0.0f,  0.0f },
    (vec3){ -1.0f,  0.0f,  0.0f },
};

const vec3 FORWARD_VECTORS[6] = {
    (vec3){  0.0f,  1.0f,  0.0f },
    (vec3){  0.0f,  1.0f,  0.0f },
    (vec3){  0.0f,  0.0f,  1.0f },
    (vec3){  0.0f,  0.0f, -1.0f },
    (vec3){  0.0f,  1.0f,  0.0f },
    (vec3){  0.0f,  1.0f,  0.0f },
};

int apple_create(game_world_t* world, mesh_t* apple_mesh, snake_t* snake)
{
    game_object_t apple;
    vec3 candidate;
    bool valid;
    int max_attempts = 200;

    do {
        valid = true;
        int vector_index = rand_int(0, 5);
        int x = rand_int(-4, 4);
        int y = rand_int(-4, 4);

        candidate = vec3_multiply_scalar(&NORMAL_VECTORS[vector_index], 5.5f);
        vec3 inward = vec3_multiply_scalar(&NORMAL_VECTORS[vector_index], -0.5f);
        candidate = vec3_add(&candidate, &inward);

        vec3 right_position   = vec3_multiply_scalar(&RIGHT_VECTORS[vector_index],   1.1f * x);
        vec3 forward_position = vec3_multiply_scalar(&FORWARD_VECTORS[vector_index], 1.1f * y);

        candidate = vec3_add(&candidate, &right_position);
        candidate = vec3_add(&candidate, &forward_position);

        // Checa colisão com cada segmento da cobra
        for (int i = 0; i < snake->segments.size; i++)
        {
            vec3* seg = vec3_array_get(&snake->segments, i);
            if (!seg) continue;

            float dx = seg->x - candidate.x;
            float dy = seg->y - candidate.y;
            float dz = seg->z - candidate.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < 1.0f)
            {
                valid = false;
                break;
            }
        }

        max_attempts--;
    } while (!valid && max_attempts > 0);

    apple = game_object_create();
    apple.transform.position = candidate;
    apple.mesh    = apple_mesh;
    apple.texture = NULL;

    int apple_id = game_world_add(world, apple);
    return apple_id;
}

// ------------------------------------------------------------------
// configuração
// ------------------------------------------------------------------
 
#define TEXT_ATLAS_W     512
#define TEXT_ATLAS_H     512
#define TEXT_FIRST_CHAR  32
#define TEXT_CHAR_COUNT  96
#define TEXT_MAX_CHARS   1024  // máximo de chars por draw call
 
// ------------------------------------------------------------------
// shaders 2D internos
// ------------------------------------------------------------------
 
static const char* TEXT_VS =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "layout(location = 1) in vec2 a_uv;\n"
    "uniform mat4 u_projection;\n"
    "uniform mat4 u_model;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = u_projection * u_model * vec4(a_pos, 0.0, 1.0);\n"
    "    v_uv = a_uv;\n"
    "}\n";
 
static const char* TEXT_FS =
    "#version 330 core\n"
    "in vec2 v_uv;\n"
    "uniform sampler2D u_atlas;\n"
    "uniform vec4 u_color;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    float alpha = texture(u_atlas, v_uv).r;\n"
    "    frag_color = vec4(u_color.rgb, u_color.a * alpha);\n"
    "}\n";
 
// ------------------------------------------------------------------
// tipos internos
// ------------------------------------------------------------------
 
typedef struct {
    float x, y;
    float u, v;
} text_vertex_t;
 
typedef struct {
    // GPU
    GLuint vao, vbo, ibo;
    GLuint shader;
    GLuint atlas_tex;
 
    // font data
    stbtt_bakedchar cdata[TEXT_CHAR_COUNT];
 
    // batch
    text_vertex_t* verts;   // TEXT_MAX_CHARS * 4 vértices
    uint32_t*      indices; // TEXT_MAX_CHARS * 6 índices
    int            quad_count;
 
    // tela
    int screen_w, screen_h;
} text_renderer_t;
 
// instância global (pode ser ponteiro se preferir)
static text_renderer_t g_text;
 
// ------------------------------------------------------------------
// helpers internos
// ------------------------------------------------------------------
 
static GLuint text_compile_shader(const char* vs_src, const char* fs_src)
{
    char log[512];
    int  ok;
 
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vs_src, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        glGetShaderInfoLog(vs, 512, NULL, log);
        printf("[text] VS error: %s\n", log);
    }
 
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fs_src, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        glGetShaderInfoLog(fs, 512, NULL, log);
        printf("[text] FS error: %s\n", log);
    }
 
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        glGetProgramInfoLog(prog, 512, NULL, log);
        printf("[text] Link error: %s\n", log);
    }
 
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}
 
static void text_set_projection(int w, int h)
{
    float L = 0.0f, R = (float)w;
    float T = 0.0f, B = (float)h;
 
    float m[16] = {
        2.0f/(R-L),    0,             0,  0,
        0,             2.0f/(T-B),    0,  0,
        0,             0,            -1,  0,
        -(R+L)/(R-L), -(T+B)/(T-B),  0,  1
    };
 
    glUseProgram(g_text.shader);
    GLint loc = glGetUniformLocation(g_text.shader, "u_projection");
    glUniformMatrix4fv(loc, 1, GL_FALSE, m);
}
 
int text_renderer_init(const char* ttf_path, float font_size, int screen_w, int screen_h)
{
    memset(&g_text, 0, sizeof(g_text));
    g_text.screen_w = screen_w;
    g_text.screen_h = screen_h;
 
    // lê o .ttf
    FILE* f = fopen(ttf_path, "rb");
    if (!f) { printf("[text] Não foi possível abrir: %s\n", ttf_path); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    unsigned char* ttf_buf = (unsigned char*)malloc(sz);
    fread(ttf_buf, 1, sz, f);
    fclose(f);
 
    // --- bake do atlas ---
    unsigned char* bitmap = (unsigned char*)malloc(TEXT_ATLAS_W * TEXT_ATLAS_H);
    int result = stbtt_BakeFontBitmap(
        ttf_buf, 0,
        font_size,
        bitmap, TEXT_ATLAS_W, TEXT_ATLAS_H,
        TEXT_FIRST_CHAR, TEXT_CHAR_COUNT,
        g_text.cdata
    );
    free(ttf_buf);
 
    if (result <= 0)
    {
        printf("[text] Aviso: atlas pode estar pequeno demais (result=%d). Aumente TEXT_ATLAS_W/H.\n", result);
    }
 
    // textura do atlas (canal R)
    glGenTextures(1, &g_text.atlas_tex);
    glBindTexture(GL_TEXTURE_2D, g_text.atlas_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                 TEXT_ATLAS_W, TEXT_ATLAS_H,
                 0, GL_RED, GL_UNSIGNED_BYTE, bitmap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    free(bitmap);
 
    g_text.shader = text_compile_shader(TEXT_VS, TEXT_FS);
 
    int max_verts   = TEXT_MAX_CHARS * 4;
    int max_indices = TEXT_MAX_CHARS * 6;
 
    g_text.verts   = (text_vertex_t*)malloc(max_verts   * sizeof(text_vertex_t));
    g_text.indices = (uint32_t*)     malloc(max_indices * sizeof(uint32_t));
 
    for (int i = 0; i < TEXT_MAX_CHARS; i++)
    {
        uint32_t b = i * 4;
        g_text.indices[i*6+0] = b+0; g_text.indices[i*6+1] = b+1; g_text.indices[i*6+2] = b+2;
        g_text.indices[i*6+3] = b+2; g_text.indices[i*6+4] = b+3; g_text.indices[i*6+5] = b+0;
    }
 
    glGenVertexArrays(1, &g_text.vao);
    glBindVertexArray(g_text.vao);
 
    glGenBuffers(1, &g_text.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_text.vbo);
    glBufferData(GL_ARRAY_BUFFER, max_verts * sizeof(text_vertex_t), NULL, GL_DYNAMIC_DRAW);
 
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(text_vertex_t), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(text_vertex_t), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
 
    glGenBuffers(1, &g_text.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_text.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_indices * sizeof(uint32_t), g_text.indices, GL_STATIC_DRAW);
 
    glBindVertexArray(0);
 
    text_set_projection(screen_w, screen_h);
 
    printf("[text] Sistema de texto inicializado: %s @ %.0fpx\n", ttf_path, font_size);
    return 1;
}
 
void text_renderer_resize(int screen_w, int screen_h)
{
    g_text.screen_w = screen_w;
    g_text.screen_h = screen_h;
    text_set_projection(screen_w, screen_h);
}
 
void text_draw(float x, float y, float r, float g, float b, float a, float scale_x, float scale_y, const char* text)
{
    float text_width;
    g_text.quad_count = 0;
    float cx = 0, cy = 0;
    
    for (const char* p = text; *p; p++)
    {
        char c = *p;
        if (c == '\n')
        {
            cy += 30.0f;
            cx = x;
            continue;
        }
        if (c < TEXT_FIRST_CHAR || c >= TEXT_FIRST_CHAR + TEXT_CHAR_COUNT) continue;
        if (g_text.quad_count >= TEXT_MAX_CHARS) break;
 
        stbtt_aligned_quad q;
        stbtt_GetBakedQuad(g_text.cdata,
                           TEXT_ATLAS_W, TEXT_ATLAS_H,
                           c - TEXT_FIRST_CHAR,
                           &cx, &cy, &q,
                           1);
 
        int qi = g_text.quad_count * 4;
        g_text.verts[qi+0] = (text_vertex_t){ q.x0, q.y0, q.s0, q.t0 };
        g_text.verts[qi+1] = (text_vertex_t){ q.x1, q.y0, q.s1, q.t0 };
        g_text.verts[qi+2] = (text_vertex_t){ q.x1, q.y1, q.s1, q.t1 };
        g_text.verts[qi+3] = (text_vertex_t){ q.x0, q.y1, q.s0, q.t1 };
        text_width = q.x1;
 
        g_text.quad_count++;
    }
 
    if (g_text.quad_count == 0) return;
 
    GLboolean depth_was_enabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blend_was_enabled = glIsEnabled(GL_BLEND);
 
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
 
    glBindBuffer(GL_ARRAY_BUFFER, g_text.vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    g_text.quad_count * 4 * sizeof(text_vertex_t),
                    g_text.verts);
 
    glUseProgram(g_text.shader);
 
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_text.atlas_tex);

    vec3 scale_vector = (vec3){ scale_x, scale_y, 1.0f };
    mat4 model = mat4_identity();
    model = mat4_translate(&model, &(vec3){ x, y, 0.0f });
    model = mat4_scale(&model, &scale_vector);
    model = mat4_translate(&model, &(vec3){ -text_width / 2.0f * scale_x, 0.0f, 0.0f });

    glUniform1i(glGetUniformLocation(g_text.shader, "u_atlas"), 0);
    glUniform4f(glGetUniformLocation(g_text.shader, "u_color"), r, g, b, a);
    glUniformMatrix4fv(glGetUniformLocation(g_text.shader, "u_model"), 1, GL_FALSE, model.m);
 
    glBindVertexArray(g_text.vao);
    glDrawElements(GL_TRIANGLES, g_text.quad_count * 6, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
 
    if (depth_was_enabled) glEnable(GL_DEPTH_TEST);
    if (!blend_was_enabled) glDisable(GL_BLEND);
}
 
void text_drawf(float x, float y, float r, float g, float b, float a, float scale_x, float scale_y, const char* fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    text_draw(x, y, r, g, b, a, scale_x, scale_y, buf);
}
 
void text_renderer_destroy()
{
    glDeleteBuffers(1, &g_text.vbo);
    glDeleteBuffers(1, &g_text.ibo);
    glDeleteVertexArrays(1, &g_text.vao);
    glDeleteTextures(1, &g_text.atlas_tex);
    glDeleteProgram(g_text.shader);
    free(g_text.verts);
    free(g_text.indices);
    memset(&g_text, 0, sizeof(g_text));
}

// ------------------------------------------------------------------
// main menu / options menu / pause menu
// ------------------------------------------------------------------

typedef enum {
    MAIN_MENU_MODE,
    OPTIONS_MODE,
    GAME_MODE,
    PAUSE_MENU_MODE
} game_mode_t;

game_mode_t game_mode = MAIN_MENU_MODE;

const vec3 SELECTED_ITEM_COLOR = (vec3){ 1.0f, 1.0f, 0.0f };
const vec3 NORMAL_ITEM_COLOR   = (vec3){ 1.0f, 1.0f, 1.0f };

int main_menu_selected_item = 0;
int main_menu_max_items = 3;
const char MAIN_MENU_OPTIONS[3][8] = { "Start", "Options", "Exit" };
vec2 main_menu_items_scale[3] = { (vec2){ 1.0f, 1.0f }, (vec2){ 1.0f, 1.0f }, (vec2){ 1.0f, 1.0f } };

int options_selected_item = 0;
int options_max_items = 2;
const char OPTIONS_OPTIONS[2][20] = { "Back", "Toggle Debug Cam" };
vec2 options_items_scale[2] = { (vec2){ 1.0f, 1.0f }, (vec2){ 1.0f, 1.0f } };

int pause_selected_item = 0;
int pause_max_items = 2;
const char PAUSE_OPTIONS[2][10] = { "Resume", "Main Menu" };
vec2 pause_items_scale[2] = { (vec2){ 1.0f, 1.0f }, (vec2){ 1.0f, 1.0f } };

// ------------------------------------------------------------------
// callbacks
// ------------------------------------------------------------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    text_renderer_resize(width, height);
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------

const int   WINDOW_WIDTH  = 1366;
const int   WINDOW_HEIGHT = 768;
const char* WINDOW_TITLE  = "Game FOG";

int main()
{
    if (!glfwInit())
    {
        printf("Erro ao inicializar GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
    if (!window)
    {
        printf("Erro ao criar janela\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        printf("Erro ao inicializar GLAD\n");
        return -1;
    }

    text_renderer_init("arial.ttf", 24.0f, WINDOW_WIDTH, WINDOW_HEIGHT);

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ------------------------------------------------------------------
    // sistemas
    // ------------------------------------------------------------------

    rand_init();
    time_init();
    input_init(window);

    shader_t shader = shader_create_from_src(vertex_src, fragment_src);

    renderer3d_t renderer;
    renderer3d_init(&renderer);

    mat4 projection = mat4_perspective(
        PI / 3.0f,
        (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT,
        0.1f, 200.0f
    );

    // ------------------------------------------------------------------
    // assets: cubo
    // ------------------------------------------------------------------

    vertex3d_t cube_verts[] =
    {
        {{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{0,0},{0,0,1},1}, {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{0,0,1},1},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,0,1},1}, {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,0,1},1},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,0,-1},1},{{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,0,-1},1},
        {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{1,1},{0,0,-1},1},{{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{0,0,-1},1},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{-1,0,0},1},{{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{-1,0,0},1},
        {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{-1,0,0},1},{{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{-1,0,0},1},
        {{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{1,0,0},1}, {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{1,0,0},1},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{1,0,0},1}, {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{1,0,0},1},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,-1,0},1},{{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,-1,0},1},
        {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,-1,0},1},{{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,-1,0},1},
        {{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,1,0},1}, {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,1,0},1},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,1,0},1}, {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,1,0},1},
    };

    vertex3d_t cube_no_tex_verts[] =
    {
        {{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{0,0},{0,0,1},0}, {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{0,0,1},0},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,0,1},0}, {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,0,1},0},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,0,-1},0},{{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,0,-1},0},
        {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{1,1},{0,0,-1},0},{{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{0,0,-1},0},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{-1,0,0},0},{{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{-1,0,0},0},
        {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{-1,0,0},0},{{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{-1,0,0},0},
        {{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{1,0,0},0}, {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,0},{1,0,0},0},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{1,0,0},0}, {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,1},{1,0,0},0},
        {{-0.5f,-0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,-1,0},0},{{ 0.5f,-0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,-1,0},0},
        {{ 0.5f,-0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,-1,0},0},{{-0.5f,-0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,-1,0},0},
        {{-0.5f, 0.5f,-0.5f},{1,1,1,0.5},{0,0},{0,1,0},0}, {{ 0.5f, 0.5f,-0.5f},{1,1,1,0.5},{1,0},{0,1,0},0},
        {{ 0.5f, 0.5f, 0.5f},{1,1,1,0.5},{1,1},{0,1,0},0}, {{-0.5f, 0.5f, 0.5f},{1,1,1,0.5},{0,1},{0,1,0},0},
    };

    uint32_t cube_indices[] =
    {
         0, 1, 2,  2, 3, 0,   4, 5, 6,  6, 7, 4,
         8, 9,10, 10,11, 8,  12,13,14, 14,15,12,
        16,17,18, 18,19,16,  20,21,22, 22,23,20,
    };

    game_world_t world;
    game_world_init(&world);

    mesh_t cube = mesh_create(cube_verts, 24, cube_indices, 36);
    mesh_t cube_no_tex = mesh_create(cube_no_tex_verts, 24, cube_indices, 36);
    texture_t cube_texture = texture_load("box.jpg");

    // ------------------------------------------------------------------
    // cria o cubo
    // ------------------------------------------------------------------

    const int CUBE_SIZE = 9;

    for (int i = -5; i <= 5; i++)
    {
        for (int j = -5; j <= 5; j++)
        {
            for (int k = -5; k <= 5; k++)
            {
                bool on_x_face = (i == -5 || i == 5);
                bool on_y_face = (j == -5 || j == 5);
                bool on_z_face = (k == -5 || k == 5);

                int face_count = (on_x_face ? 1 : 0) + (on_y_face ? 1 : 0) + (on_z_face ? 1 : 0);
                if (face_count != 1) continue;

                game_object_t block = game_object_create();
                block.mesh = &cube_no_tex;
                block.texture = NULL;
                block.blend_color = (vec3){ 1.0, 0.7, 0.5 };

                vec3 scale = (vec3){ 1.0, 1.0, 1.0 };
                vec3 block_position;
                if (on_x_face)
                {
                    scale.x = 0.1f;
                    block_position = (vec3){ 1.1f * i - sign(i) * 0.5f, 1.1f * j, 1.1f * k };
                }
                
                if (on_y_face)
                {
                    scale.y = 0.1f;
                    block_position = (vec3){ 1.1f * i, 1.1f * j - sign(j) * 0.5f, 1.1f * k };
                }

                if (on_z_face)
                {
                    scale.z = 0.1f;
                    block_position = (vec3){ 1.1f * i, 1.1f * j, 1.1f * k - sign(k) * 0.5f };
                }
                
                block.transform.scale = scale;

                block.transform.position = vec3_subtract(&block_position, &(vec3){ 0.0f, 0.0f, 0.5f });

                game_world_add(&world, block);
            }
        }
    }

    // ------------------------------------------------------------------
    // game world
    // ------------------------------------------------------------------

    snake_t snake;
    snake_init(&snake, 10.0f, 0.15f);
    snake_grow(&snake);
    snake_grow(&snake);
    snake_grow(&snake);

    int apple_id = apple_create(&world, &cube_no_tex, &snake);
    game_object_t* apple = game_world_get_object(&world, apple_id);

    // ------------------------------------------------------------------
    // câmera debug
    // ------------------------------------------------------------------

    camera debug_cam;
    debug_cam.position      = (vec3){ 0.0f, 6.0f, 12.0f };
    debug_cam.last_position = debug_cam.position;
    debug_cam.yaw           = -PI / 2.0f;
    debug_cam.last_yaw      = debug_cam.yaw;
    debug_cam.pitch         = -0.3f;
    debug_cam.last_pitch    = debug_cam.pitch;
    debug_cam.speed         = 8.0f;
    debug_cam.sensitivity   = 0.002f;

    camera game_camera;
    game_camera.position = (vec3){ 0.0f, 6.0f, 12.0f };

    static double debug_last_x    = 0.0, debug_last_y = 0.0;
    static int    debug_first_mouse = 1;

    // ------------------------------------------------------------------
    // loop principal
    // ------------------------------------------------------------------

    while (!glfwWindowShouldClose(window))
    {
        time_update();
        input_update();
        float dt = time_delta();

        switch (game_mode)
        {
            case MAIN_MENU_MODE:
            {
                glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                text_draw(
                    WINDOW_WIDTH / 2.0f, WINDOW_HEIGHT / 2.0f - 120.0f,
                    1, 1, 1, 1, 2.0f, 2.0f, "SNAKE 3D"
                );

                for (int i = 0; i < main_menu_max_items; i++)
                {
                    vec3 color = main_menu_selected_item == i ? SELECTED_ITEM_COLOR : NORMAL_ITEM_COLOR;
                    main_menu_items_scale[i] = vec2_lerp(
                        &main_menu_items_scale[i], &(vec2){ 1.0f, 1.0f }, 0.075f
                    );
                    text_draw(
                        WINDOW_WIDTH / 2.0f,
                        WINDOW_HEIGHT / 2.0f + (i - 1) * 50.0f,
                        color.x, color.y, color.z, 1.0f,
                        main_menu_items_scale[i].x, main_menu_items_scale[i].y,
                        MAIN_MENU_OPTIONS[i]
                    );
                }

                if (input_get_key_down(GLFW_KEY_W))
                {
                    main_menu_selected_item--;
                    if (main_menu_selected_item < 0)
                        main_menu_selected_item = main_menu_max_items - 1;
                    main_menu_items_scale[main_menu_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_S))
                {
                    main_menu_selected_item++;
                    if (main_menu_selected_item > main_menu_max_items - 1)
                        main_menu_selected_item = 0;
                    main_menu_items_scale[main_menu_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_ENTER))
                {
                    switch (main_menu_selected_item)
                    {
                        case 0:
                            snake_free(&snake);
                            snake_init(&snake, 10.0f, 0.15f);
                            snake_grow(&snake);
                            snake_grow(&snake);
                            snake_grow(&snake);
                            game_world_remove(&world, apple_id);
                            apple_id = apple_create(&world, &cube_no_tex, &snake);
                            apple = game_world_get_object(&world, apple_id);
                            game_mode = GAME_MODE;
                        break;

                        case 1:
                            options_selected_item = 0;
                            game_mode = OPTIONS_MODE;
                        break;

                        case 2:
                            glfwSetWindowShouldClose(window, 1);
                        break;
                    }
                }
            }
            break;

            case OPTIONS_MODE:
            {
                glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                text_draw(
                    WINDOW_WIDTH / 2.0f, WINDOW_HEIGHT / 2.0f - 100.0f,
                    1, 1, 1, 1, 1.5f, 1.5f, "OPTIONS"
                );

                char debug_label[32];
                snprintf(debug_label, sizeof(debug_label), "Debug Cam: %s", camera_free ? "ON" : "OFF");

                const char* options_labels[2] = { "Back", debug_label };

                for (int i = 0; i < options_max_items; i++)
                {
                    vec3 color = options_selected_item == i ? SELECTED_ITEM_COLOR : NORMAL_ITEM_COLOR;
                    options_items_scale[i] = vec2_lerp(
                        &options_items_scale[i], &(vec2){ 1.0f, 1.0f }, 0.075f
                    );
                    text_draw(
                        WINDOW_WIDTH / 2.0f,
                        WINDOW_HEIGHT / 2.0f + (i - 1) * 50.0f,
                        color.x, color.y, color.z, 1.0f,
                        options_items_scale[i].x, options_items_scale[i].y,
                        options_labels[i]
                    );
                }

                if (input_get_key_down(GLFW_KEY_W))
                {
                    options_selected_item--;
                    if (options_selected_item < 0)
                        options_selected_item = options_max_items - 1;
                    options_items_scale[options_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_S))
                {
                    options_selected_item++;
                    if (options_selected_item > options_max_items - 1)
                        options_selected_item = 0;
                    options_items_scale[options_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_ENTER))
                {
                    switch (options_selected_item)
                    {
                        case 0:
                            game_mode = MAIN_MENU_MODE;
                        break;

                        case 1:
                            camera_free = !camera_free;
                            options_items_scale[1] = (vec2){ 1.3f, 1.3f };
                        break;
                    }
                }

                if (input_get_key_down(GLFW_KEY_ESCAPE))
                    game_mode = MAIN_MENU_MODE;
            }
            break;

            case GAME_MODE:
            {
                if (input_get_key_down(GLFW_KEY_ESCAPE))
                {
                    pause_selected_item = 0;
                    pause_items_scale[0] = (vec2){ 1.0f, 1.0f };
                    pause_items_scale[1] = (vec2){ 1.0f, 1.0f };
                    game_mode = PAUSE_MENU_MODE;
                    break;
                }

                snake_update(&snake, time_total());

                vec3* head = vec3_array_get(&snake.segments, 0);
                if (head && apple)
                {
                    float dx = apple->transform.position.x - head->x;
                    float dy = apple->transform.position.y - head->y;
                    float dz = apple->transform.position.z - head->z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < 1.2f)
                    {
                        game_world_remove(&world, apple_id);
                        snake_grow(&snake);
                        apple_id = apple_create(&world, &cube_no_tex, &snake);
                        apple = game_world_get_object(&world, apple_id);
                    }
                }

                if (input_get_key_down(GLFW_KEY_CAPS_LOCK))
                    snake.paused = !snake.paused;

                if (input_get_key_down(GLFW_KEY_TAB))
                {
                    camera_free = !camera_free;
                    if (camera_free)
                    {
                        debug_cam.last_position = debug_cam.position;
                        debug_first_mouse = 1;
                    }
                }

                game_world_update(&world, dt);

                mat4 view;

                if (camera_free)
                {
                    double mouse_x, mouse_y;
                    input_get_mouse_position(&mouse_x, &mouse_y);

                    if (debug_first_mouse)
                    {
                        debug_last_x = mouse_x;
                        debug_last_y = mouse_y;
                        debug_first_mouse = 0;
                    }

                    float dx = (float)(mouse_x - debug_last_x) * debug_cam.sensitivity;
                    float dy = (float)(debug_last_y - mouse_y)  * debug_cam.sensitivity;
                    debug_last_x = mouse_x;
                    debug_last_y = mouse_y;

                    debug_cam.yaw   += dx;
                    debug_cam.pitch += dy;
                    if (debug_cam.pitch >  1.5f) debug_cam.pitch =  1.5f;
                    if (debug_cam.pitch < -1.5f) debug_cam.pitch = -1.5f;

                    vec3  fwd   = camera_get_forward(&debug_cam);
                    vec3  right = camera_get_right(&debug_cam);
                    float vel   = debug_cam.speed * dt;

                    if (input_get_key(GLFW_KEY_W))
                        debug_cam.position = vec3_add(&debug_cam.position, &(vec3){ fwd.x*vel, fwd.y*vel, fwd.z*vel });
                    if (input_get_key(GLFW_KEY_S))
                        debug_cam.position = vec3_subtract(&debug_cam.position, &(vec3){ fwd.x*vel, fwd.y*vel, fwd.z*vel });
                    if (input_get_key(GLFW_KEY_A))
                        debug_cam.position = vec3_subtract(&debug_cam.position, &(vec3){ right.x*vel, right.y*vel, right.z*vel });
                    if (input_get_key(GLFW_KEY_D))
                        debug_cam.position = vec3_add(&debug_cam.position, &(vec3){ right.x*vel, right.y*vel, right.z*vel });
                    if (input_get_key(GLFW_KEY_Q))
                        debug_cam.position = vec3_add(&debug_cam.position, &(vec3){ 0.0f, -vel, 0.0f });
                    if (input_get_key(GLFW_KEY_E))
                        debug_cam.position = vec3_add(&debug_cam.position, &(vec3){ 0.0f,  vel, 0.0f });

                    vec3 target = vec3_add(&debug_cam.position, &fwd);
                    view = mat4_look_at(debug_cam.position, target, CAMERA_UP);
                }
                else
                {
                    vec3* snake_head_pos = vec3_array_get(&snake.segments, 0);
                    if (!snake_head_pos)
                    {
                        view = mat4_look_at(game_camera.position, vec3_zero(), CAMERA_UP);
                    }
                    else
                    {
                        vec3 behind = vec3_negate(&snake.head_forward);
                        behind = vec3_normalize(&behind);

                        vec3 camera_offset = vec3_multiply_scalar(&behind, 8.0f);
                        vec3 up_offset     = vec3_multiply_scalar(&snake.current_face_normal, 4.0f);
                        vec3 target_pos    = vec3_add(snake_head_pos, &camera_offset);
                        target_pos         = vec3_add(&target_pos, &up_offset);

                        game_camera.position = vec3_lerp(&game_camera.position, &target_pos, 0.08f);

                        vec3 look_target = *snake_head_pos;
                        static vec3 smooth_target = {0};
                        smooth_target = vec3_lerp(&smooth_target, &look_target, 0.15f);

                        view = mat4_look_at(game_camera.position, smooth_target, snake.current_face_normal);
                    }
                }

                glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                game_world_render(&world, shader.id, &view, &projection);
                snake_render(&snake, &cube, &cube, shader.id, &cube_texture, &view, &projection);

                renderer3d_begin_batch(&renderer, shader.id, &view, &projection);
                renderer3d_end_batch(&renderer);

                int score = snake.segments.size - 4; // desconta os 3 iniciais + cabeça
                if (score < 0) score = 0;
                text_drawf(WINDOW_WIDTH / 2.0f, 40,  1, 1, 1, 1, 1.0f, 1.0f, "Score: %d", score);
            }
            break;

            case PAUSE_MENU_MODE:
            {
                mat4 view;
                {
                    vec3* snake_head_pos = vec3_array_get(&snake.segments, 0);
                    if (!snake_head_pos)
                    {
                        view = mat4_look_at(game_camera.position, vec3_zero(), CAMERA_UP);
                    }
                    else
                    {
                        vec3 look_target = *snake_head_pos;
                        view = mat4_look_at(game_camera.position, look_target, snake.current_face_normal);
                    }
                }

                glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                game_world_render(&world, shader.id, &view, &projection);
                snake_render(&snake, &cube, &cube, shader.id, &cube_texture, &view, &projection);

                renderer3d_begin_batch(&renderer, shader.id, &view, &projection);
                renderer3d_end_batch(&renderer);

                text_draw(
                    WINDOW_WIDTH / 2.0f, WINDOW_HEIGHT / 2.0f - 80.0f,
                    1, 1, 0.3f, 1, 1.8f, 1.8f, "PAUSED"
                );

                for (int i = 0; i < pause_max_items; i++)
                {
                    vec3 color = pause_selected_item == i ? SELECTED_ITEM_COLOR : NORMAL_ITEM_COLOR;
                    pause_items_scale[i] = vec2_lerp(
                        &pause_items_scale[i], &(vec2){ 1.0f, 1.0f }, 0.1f
                    );
                    text_draw(
                        WINDOW_WIDTH / 2.0f,
                        WINDOW_HEIGHT / 2.0f + (i - 0) * 50.0f,
                        color.x, color.y, color.z, 1.0f,
                        pause_items_scale[i].x, pause_items_scale[i].y,
                        PAUSE_OPTIONS[i]
                    );
                }

                if (input_get_key_down(GLFW_KEY_W))
                {
                    pause_selected_item--;
                    if (pause_selected_item < 0)
                        pause_selected_item = pause_max_items - 1;
                    pause_items_scale[pause_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_S))
                {
                    pause_selected_item++;
                    if (pause_selected_item > pause_max_items - 1)
                        pause_selected_item = 0;
                    pause_items_scale[pause_selected_item] = (vec2){ 1.5f, 1.5f };
                }

                if (input_get_key_down(GLFW_KEY_ENTER))
                {
                    switch (pause_selected_item)
                    {
                        case 0:
                            game_mode = GAME_MODE;
                        break;

                        case 1:
                            game_mode = MAIN_MENU_MODE;
                            main_menu_selected_item = 0;
                        break;
                    }
                }

                if (input_get_key_down(GLFW_KEY_ESCAPE))
                    game_mode = GAME_MODE;
            }
            break;
        }
        

        

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ------------------------------------------------------------------
    // cleanup
    // ------------------------------------------------------------------

    renderer3d_destroy(&renderer);
    snake_free(&snake);
    game_object_array_free(&(world.objects));
    text_renderer_destroy();
    glfwTerminate();
    return 0;
}