#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f

const char* vertex_src =
    "#version 330 core\n"
    "layout(location = 0) in vec3 a_position;\n"
    "layout(location = 1) in vec4 a_color;\n"
    "layout(location = 2) in vec2 a_tex_coord;\n"
    "layout(location = 3) in float a_tex_index;\n"
    "uniform mat4 u_model;\n"
    "uniform mat4 u_view;\n"
    "uniform mat4 u_projection;\n"
    "out vec4 v_color;\n"
    "out vec2 v_tex_coord;\n"
    "out float v_tex_index;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);\n"
    "    v_color     = a_color;\n"
    "    v_tex_coord = a_tex_coord;\n"
    "    v_tex_index = a_tex_index;\n"
    "}\n";

const char* fragment_src =
    "#version 330 core\n"
    "in vec4  v_color;\n"
    "in vec2  v_tex_coord;\n"
    "in float v_tex_index;\n"
    "uniform sampler2D u_textures[16];\n"
    "out vec4 frag_color;\n"
    "void main()\n"
    "{\n"
    "    vec4 tex_color;\n"
    "    int idx = int(v_tex_index);\n"
    "    if      (idx == 0)  tex_color = texture(u_textures[0],  v_tex_coord);\n"
    "    else if (idx == 1)  tex_color = texture(u_textures[1],  v_tex_coord);\n"
    "    else if (idx == 2)  tex_color = texture(u_textures[2],  v_tex_coord);\n"
    "    else if (idx == 3)  tex_color = texture(u_textures[3],  v_tex_coord);\n"
    "    else if (idx == 4)  tex_color = texture(u_textures[4],  v_tex_coord);\n"
    "    else if (idx == 5)  tex_color = texture(u_textures[5],  v_tex_coord);\n"
    "    else if (idx == 6)  tex_color = texture(u_textures[6],  v_tex_coord);\n"
    "    else if (idx == 7)  tex_color = texture(u_textures[7],  v_tex_coord);\n"
    "    else if (idx == 8)  tex_color = texture(u_textures[8],  v_tex_coord);\n"
    "    else if (idx == 9)  tex_color = texture(u_textures[9],  v_tex_coord);\n"
    "    else if (idx == 10) tex_color = texture(u_textures[10], v_tex_coord);\n"
    "    else if (idx == 11) tex_color = texture(u_textures[11], v_tex_coord);\n"
    "    else if (idx == 12) tex_color = texture(u_textures[12], v_tex_coord);\n"
    "    else if (idx == 13) tex_color = texture(u_textures[13], v_tex_coord);\n"
    "    else if (idx == 14) tex_color = texture(u_textures[14], v_tex_coord);\n"
    "    else                tex_color = texture(u_textures[15], v_tex_coord);\n"
    "    frag_color = tex_color * v_color;\n"
    "}\n";

// ------------------------------------------------------------------
// tipos
// ------------------------------------------------------------------

typedef struct { float x, y, z;       } vec3;
typedef struct { float x, y, z, w;    } vec4;
typedef struct { float x, y;          } vec2;
typedef struct { float m[16];         } mat4;

// ------------------------------------------------------------------
// declarações
// ------------------------------------------------------------------

vec3  vec3_add(vec3* a, vec3* b);
vec3  vec3_subtract(vec3* a, vec3* b);
vec3  vec3_cross(vec3* a, vec3* b);
float vec3_dot(vec3* a, vec3* b);

mat4 mat4_identity();
mat4 mat4_multiply(mat4* a, mat4* b);
mat4 mat4_translate(mat4* mat, vec3* vec);
mat4 mat4_rotate(mat4* mat, vec3* axis, float angle);
mat4 mat4_scale(mat4* mat, vec3* vec);
void mat4_print(mat4* mat);
mat4 mat4_perspective(float fov, float aspect,
                              float near, float far);

mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up);

// ------------------------------------------------------------------
// math interno (renderer)
// ------------------------------------------------------------------

static void mat4_identity_raw(float out[16])
{
    memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

// FIX: corrigido para column-major (j*4+i ao invés de i*4+j)
static void mat4_mul_raw(float out[16], float a[16], float b[16])
{
    float tmp[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                tmp[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
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
    out[5]  =  cosf(a);
    out[9]  = -sinf(a);
    out[6]  =  sinf(a);
    out[10] =  cosf(a);
}

static void build_rotate_y(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0]  =  cosf(a);
    out[8]  =  sinf(a);
    out[2]  = -sinf(a);
    out[10] =  cosf(a);
}

static void build_rotate_z(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0]  =  cosf(a);
    out[4]  = -sinf(a);
    out[1]  =  sinf(a);
    out[5]  =  cosf(a);
}

static void build_transform(float out[16], vec3 pos, vec3 rot, float sx, float sy)
{
    float t[16], rx[16], ry[16], rz[16], s[16], tmp[16];
    build_translate(t,  pos);
    build_rotate_x (rx, rot.x);
    build_rotate_y (ry, rot.y);
    build_rotate_z (rz, rot.z);
    build_scale    (s,  sx, sy, 1.0f);

    mat4_mul_raw(tmp, rz, ry);
    mat4_mul_raw(tmp, tmp, rx);
    mat4_mul_raw(tmp, tmp, s);
    mat4_mul_raw(out, t,   tmp);
}

static vec3 transform_vec4(float m[16], vec4 v)
{
    return (vec3)
    {
        m[0]*v.x + m[4]*v.y + m[8] *v.z + m[12]*v.w,
        m[1]*v.x + m[5]*v.y + m[9] *v.z + m[13]*v.w,
        m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
    };
}

// ------------------------------------------------------------------
// camera helpers
// ------------------------------------------------------------------



// ------------------------------------------------------------------
// input
// ------------------------------------------------------------------

#define MAX_KEYS 512

static GLFWwindow* m_window       = NULL;
static int         m_current_keys[MAX_KEYS];
static int         m_previous_keys[MAX_KEYS];

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

int input_get_key_down(int key) { return  m_current_keys[key] && !m_previous_keys[key]; }
int input_get_key_up(int key)   { return !m_current_keys[key] &&  m_previous_keys[key]; }

int input_get_mouse_button(int button)
{
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

void input_get_mouse_position(double* x, double* y)
{
    glfwGetCursorPos(m_window, x, y);
}

// ------------------------------------------------------------------
// time
// ------------------------------------------------------------------

static double last_time = 0.0;
static float  delta     = 0.0f;

void time_init()
{
    last_time = glfwGetTime();
    delta     = 0.0f;
}

void time_update()
{
    double now = glfwGetTime();
    delta      = (float)(now - last_time);
    last_time  = now;
}

float time_total()
{
    return (float)glfwGetTime();
}

float time_delta()
{
    return delta;
}

// ------------------------------------------------------------------
// shader
// ------------------------------------------------------------------

typedef struct
{
    unsigned int id;
} shader_t;

static GLint get_uniform_location(GLuint program, const char* name)
{
    GLint location = glGetUniformLocation(program, name);
    if (location == -1)
        printf("Warning: uniform '%s' not found.\n", name);
    return location;
}

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

static void set_uniform_int_array(GLuint prog, const char* name,
                                  int* values, int count)
{
    glUniform1iv(uniform_loc(prog, name), count, values);
}

static char* read_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(size + 1);
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
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
        glGetShaderInfoLog(vertex, 512, NULL, info_log); printf("ERROR::VERTEX\n%s\n", info_log);
    }

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fs, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment, 512, NULL, info_log); printf("ERROR::FRAGMENT\n%s\n", info_log);
    }

    shader.id = glCreateProgram();
    glAttachShader(shader.id, vertex);
    glAttachShader(shader.id, fragment);
    glLinkProgram(shader.id);
    glGetProgramiv(shader.id, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shader.id, 512, NULL, info_log); printf("ERROR::LINKING\n%s\n", info_log);
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return shader;
}

shader_t shader_create(const char* vertex_path, const char* fragment_path)
{
    char* vs = read_file(vertex_path);
    char* fs = read_file(fragment_path);
    if (!vs || !fs) { free(vs); free(fs); return (shader_t){0}; }
    shader_t s = compile_shader(vs, fs);
    free(vs); free(fs);
    return s;
}

shader_t shader_create_from_src(const char* vs, const char* fs)
{
    return compile_shader(vs, fs);
}

void shader_use(shader_t* shader)
{
    glUseProgram(shader->id);
}
void shader_set_bool(shader_t* s, const char* name, int value)
{
    glUniform1i(get_uniform_location(s->id, name), value);
}
void shader_set_int(shader_t* s, const char* name, int value)
{
    glUniform1i(get_uniform_location(s->id, name), value);
}
void shader_set_float(shader_t* s, const char* name, float value)
{
    glUniform1f(get_uniform_location(s->id, name), value);
}
void shader_set_int_array(shader_t* s, const char* name, int* values, unsigned int count)
{
    glUniform1iv(get_uniform_location(s->id, name), count, values);
}
void shader_set_mat4(shader_t* s, const char* name, mat4* mat)
{
    glUniformMatrix4fv(get_uniform_location(s->id, name), 1, GL_FALSE, mat->m);
}

// ------------------------------------------------------------------
// vertex / mesh
// ------------------------------------------------------------------

typedef struct
{
    vec3  position;
    vec4  color;
    vec2  tex_coord;
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
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, tex_index));
    glEnableVertexAttribArray(3);
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

    // FIX: cache para reuso no overflow do batch
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
        indices[i+0] = offset + 0;
        indices[i+1] = offset + 1;
        indices[i+2] = offset + 2;
        indices[i+3] = offset + 2;
        indices[i+4] = offset + 3;
        indices[i+5] = offset + 0;
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
    for (uint32_t i = 0; i < r->texture_slot_index; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, r->texture_slots[i]);
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
    r->texture_slot_index = 1;

    // FIX: cacheia para uso no overflow
    r->current_shader     = shader_id;
    r->current_view       = *view;
    r->current_projection = *projection;

    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_view",       view);
    set_uniform_mat4(shader_id, "u_projection", projection);

    // FIX: seta u_model como identidade — transform já está nos vértices
    mat4 identity = mat4_identity();
    set_uniform_mat4(shader_id, "u_model", &identity);

    int samplers[MAX_TEXTURES];
    for (int i = 0; i < MAX_TEXTURES; i++) samplers[i] = i;
    set_uniform_int_array(shader_id, "u_textures", samplers, MAX_TEXTURES);
}

void renderer3d_end_batch(renderer3d_t* r)
{
    ptrdiff_t size =
        (uint8_t*)r->vertex_buffer_ptr -
        (uint8_t*)r->vertex_buffer_base;

    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, r->vertex_buffer_base);
    renderer3d_flush(r);
}

void renderer3d_draw_quad(renderer3d_t* r,
                           vec3 position, vec3 rotation, vec2 size,
                           GLuint texture_id, vec4 color)
{
    // FIX: usa cache ao invés de passar NULL no overflow
    if (r->index_count >= MAX_INDICES)
    {
        renderer3d_end_batch(r);
        renderer3d_begin_batch(r, r->current_shader,
                               &r->current_view,
                               &r->current_projection);
    }

    float tex_index = 0.0f;
    for (uint32_t i = 1; i < r->texture_slot_index; i++)
    {
        if (r->texture_slots[i] == texture_id)
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
                                   &r->current_view,
                                   &r->current_projection);
        }
        tex_index = (float)r->texture_slot_index;
        r->texture_slots[r->texture_slot_index++] = texture_id;
    }

    float transform[16];
    build_transform(transform, position, rotation, size.x, size.y);

    vec2 tex_coords[4] = {
        { 0.0f, 0.0f }, { 1.0f, 0.0f },
        { 1.0f, 1.0f }, { 0.0f, 1.0f }
    };

    for (int i = 0; i < 4; i++)
    {
        r->vertex_buffer_ptr->position  = transform_vec4(transform, r->quad_vertex_positions[i]);
        r->vertex_buffer_ptr->color     = color;
        r->vertex_buffer_ptr->tex_coord = tex_coords[i];
        r->vertex_buffer_ptr->tex_index = tex_index;
        r->vertex_buffer_ptr++;
    }

    r->index_count += 6;
}

void renderer3d_draw_mesh(mesh_t* mesh, GLuint shader_id, mat4* model,
                           mat4* view, mat4* projection)
{
    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_model",      model);
    set_uniform_mat4(shader_id, "u_view",       view);
    set_uniform_mat4(shader_id, "u_projection", projection);

    // textura slot 0 ativa para meshes
    int samplers[MAX_TEXTURES];
    for (int i = 0; i < MAX_TEXTURES; i++) samplers[i] = i;
    set_uniform_int_array(shader_id, "u_textures", samplers, MAX_TEXTURES);

    mesh_draw(mesh);
}

// ------------------------------------------------------------------
// vec3
// ------------------------------------------------------------------

vec3 vec3_add(vec3* a, vec3* b)
{
    return (vec3){ a->x+b->x, a->y+b->y, a->z+b->z };
}

vec3 vec3_subtract(vec3* a, vec3* b)
{
    return (vec3){ a->x-b->x, a->y-b->y, a->z-b->z };
}

vec3 vec3_multiply_scalar(vec3* vec, float scalar)
{
    return (vec3){vec->x * scalar, vec->y * scalar, vec->z * scalar};
}

vec3 vec3_cross(vec3* a, vec3* b)
{
    return (vec3){
        a->y*b->z - a->z*b->y,
        a->z*b->x - a->x*b->z,
        a->x*b->y - a->y*b->x
    };
}

float vec3_dot(vec3* a, vec3* b)
{
    return a->x*b->x + a->y*b->y + a->z*b->z;
}

// ------------------------------------------------------------------
// mat4
// ------------------------------------------------------------------

// FIX: usa designated initializer .m
mat4 mat4_identity()
{
    return (mat4){ .m = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    }};
}

// FIX: corrigido para column-major
mat4 mat4_multiply(mat4* a, mat4* b)
{
    mat4 result = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                result.m[j * 4 + i] += a->m[k * 4 + i] * b->m[j * 4 + k];
    return result;
}

// FIX: índices corretos para column-major (12,13,14)
mat4 mat4_translate(mat4* mat, vec3* vec)
{
    mat4 result = *mat;
    result.m[12] += vec->x;
    result.m[13] += vec->y;
    result.m[14] += vec->z;
    return result;
}

// FIX: implementado (antes retornava UB)
mat4 mat4_rotate(mat4* mat, vec3* axis, float angle)
{
    float x = axis->x, y = axis->y, z = axis->z;
    float len = sqrtf(x*x + y*y + z*z);
    x /= len; y /= len; z /= len;

    float c = cosf(angle), s = sinf(angle), t = 1.0f - c;

    float r[16];
    mat4_identity_raw(r);
    r[0]  = t*x*x + c;   r[4]  = t*x*y - s*z; r[8]  = t*x*z + s*y;
    r[1]  = t*x*y + s*z; r[5]  = t*y*y + c;   r[9]  = t*y*z - s*x;
    r[2]  = t*x*z - s*y; r[6]  = t*y*z + s*x; r[10] = t*z*z + c;

    mat4 rot = {0}; memcpy(rot.m, r, 64);
    return mat4_multiply(mat, &rot);
}

// FIX: implementado (antes retornava UB)
mat4 mat4_scale(mat4* mat, vec3* vec)
{
    mat4 result = *mat;
    result.m[0]  *= vec->x;
    result.m[5]  *= vec->y;
    result.m[10] *= vec->z;
    return result;
}

void mat4_print(mat4* mat)
{
    for (int i = 0; i < 4; i++)
    {
        printf("[ ");
        for (int j = 0; j < 4; j++)
            printf("%8.3f ", mat->m[i * 4 + j]);
        printf("]\n");
    }
}

mat4 mat4_perspective(float fov, float aspect,
                      float near, float far)
{
    mat4 result;
    memset(result.m, 0, 16 * sizeof(float));

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
    memset(result.m, 0, 16 * sizeof(float));

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
        f.y*up.z - f.z*up.y,
        f.z*up.x - f.x*up.z,
        f.x*up.y - f.y*up.x
    };
    float sl = sqrtf(s.x*s.x + s.y*s.y + s.z*s.z);
    s.x /= sl; s.y /= sl; s.z /= sl;

    vec3 u =
    {
        s.y*f.z - s.z*f.y,
        s.z*f.x - s.x*f.z,
        s.x*f.y - s.y*f.x
    };

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

const vec3 CAMERA_UP = (vec3){0, 1, 0};
bool camera_free = false;

typedef struct
{
    vec3 position;
    vec3 last_position; // usado para reverter o 'free movement' da camera

    float yaw;
    float last_yaw;

    float pitch;
    float last_pitch;

    float speed;
    float sensitivity;

} camera;

vec3 camera_get_forward(camera* cam)
{
    vec3 dir;

    dir.x = cosf(cam->yaw) * cosf(cam->pitch);
    dir.y = sinf(cam->pitch);
    dir.z = sinf(cam->yaw) * cosf(cam->pitch);

    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;

    return dir;
}

vec3 camera_get_right(camera* cam)
{
    vec3 forward = camera_get_forward(cam);
    vec3 right = vec3_cross(&forward, (vec3*)&CAMERA_UP);

    float len = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= len; right.y /= len; right.z /= len;

    return right;
}

// ------------------------------------------------------------------
// callbacks
// ------------------------------------------------------------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------

const int   WINDOW_WIDTH  = 1366;
const int   WINDOW_HEIGHT = 768;
const char* WINDOW_TITLE  = "Game FOG";



void mesh_draw_lines(mesh_t* mesh)
{
    glBindVertexArray(mesh->vao);
    glDrawElements(GL_LINES, mesh->index_count, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

#define GRID_SIZE 10
#define GRID_SPACING 1.0f

int main()
{
    if (!glfwInit()) { printf("Erro ao inicializar GLFW\n"); return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
    if (!window) { printf("Erro ao criar janela\n"); glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    { printf("Erro ao inicializar GLAD\n"); return -1; }

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // FIX: depth test habilitado
    glEnable(GL_DEPTH_TEST);

    time_init();
    input_init(window);

    shader_t shader = shader_create_from_src(vertex_src, fragment_src);

    vertex3d_t cube_verts[] = {
        {{ -0.5f, -0.5f,  0.5f }, { 1,0,0,1 }, { 0,0 }, 0},
        {{  0.5f, -0.5f,  0.5f }, { 1,0,0,1 }, { 1,0 }, 0},
        {{  0.5f,  0.5f,  0.5f }, { 1,0,0,1 }, { 1,1 }, 0},
        {{ -0.5f,  0.5f,  0.5f }, { 1,0,0,1 }, { 0,1 }, 0},
        {{ -0.5f, -0.5f, -0.5f }, { 0,1,0,1 }, { 0,0 }, 0},
        {{  0.5f, -0.5f, -0.5f }, { 0,1,0,1 }, { 1,0 }, 0},
        {{  0.5f,  0.5f, -0.5f }, { 0,1,0,1 }, { 1,1 }, 0},
        {{ -0.5f,  0.5f, -0.5f }, { 0,1,0,1 }, { 0,1 }, 0},
    };

    uint32_t cube_indices[] = {
        0,1,2, 2,3,0,
        4,5,6, 6,7,4,
        0,4,7, 7,3,0,
        1,5,6, 6,2,1,
        3,2,6, 6,7,3,
        0,1,5, 5,4,0
    };

    mesh_t cube = mesh_create(cube_verts, 8, cube_indices, 36);

    vertex3d_t grid_vertices[(GRID_SIZE * 2 + 1) * 4];
    uint32_t grid_indices[(GRID_SIZE * 2 + 1) * 4];

    int v = 0;
    int i = 0;

    for (int x = -GRID_SIZE; x <= GRID_SIZE; x++)
    {
        // linha paralela ao eixo Z
        grid_vertices[v++] = (vertex3d_t){ {x, 0, -GRID_SIZE}, {1,1,1,1}, {0,0}, 0 };
        grid_vertices[v++] = (vertex3d_t){ {x, 0,  GRID_SIZE}, {1,1,1,1}, {0,0}, 0 };

        grid_indices[i++] = v - 2;
        grid_indices[i++] = v - 1;

        // linha paralela ao eixo X
        grid_vertices[v++] = (vertex3d_t){ {-GRID_SIZE, 0, x}, {1,1,1,1}, {0,0}, 0 };
        grid_vertices[v++] = (vertex3d_t){ { GRID_SIZE, 0, x}, {1,1,1,1}, {0,0}, 0 };

        grid_indices[i++] = v - 2;
        grid_indices[i++] = v - 1;
    }
    mesh_t grid = mesh_create(grid_vertices, v, grid_indices, i);
    glLineWidth(10.0);

    renderer3d_t renderer;
    renderer3d_init(&renderer);

    mat4 projection = mat4_perspective(PI / 3.0f, (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 100.0f);
    mat4 view = (mat4){0};

    GLuint white_tex;
    unsigned char white[] = { 255, 255, 255, 255 };
    glGenTextures(1, &white_tex);
    glBindTexture(GL_TEXTURE_2D, white_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float cube_angle = 0.0f;

    camera camera;

    camera.position = (vec3){0, 2.0f, 3.0f};
    camera.last_position = camera.position;

    camera.yaw = -PI / 2.0f;
    camera.last_yaw = camera.yaw;

    camera.pitch = -PI / 8;
    camera.last_pitch = camera.pitch;

    camera.speed = 5.0f;
    camera.sensitivity = 0.002f;


    while (!glfwWindowShouldClose(window))
    {
        time_update();
        input_update();

        if (input_get_key(GLFW_KEY_ESCAPE))
            glfwSetWindowShouldClose(window, 1);

        cube_angle += 1.0f * time_delta();

        vec3 forward = camera_get_forward(&camera);
        vec3 right   = camera_get_right(&camera);

        if (input_get_key_down(GLFW_KEY_TAB))
        {
            camera_free = !camera_free;

            if (!camera_free)
            {
                camera.position = camera.last_position;
                camera.yaw = camera.last_yaw;
                camera.pitch = camera.last_pitch;
            }
            else
            {
                camera.last_position = camera.position;
                camera.last_yaw = camera.yaw;
                camera.last_pitch = camera.pitch;
            }
        }

        if (camera_free)
        {
            static double last_x = 0, last_y = 0;
            static int first_mouse = 1;

            double mouse_x, mouse_y;
            input_get_mouse_position(&mouse_x, &mouse_y);

            if (first_mouse)
            {
                last_x = mouse_x;
                last_y = mouse_y;
                first_mouse = 0;
            }

            float dx = mouse_x - last_x;
            float dy = last_y - mouse_y;

            last_x = mouse_x;
            last_y = mouse_y;

            dx *= camera.sensitivity;
            dy *= camera.sensitivity;

            camera.yaw   += dx;
            camera.pitch += dy;

            // clamp pitch
            if (camera.pitch >  1.5f) camera.pitch =  1.5f;
            if (camera.pitch < -1.5f) camera.pitch = -1.5f;

            float velocity = camera.speed * time_delta();

            if (input_get_key(GLFW_KEY_W))
                camera.position = vec3_add(&camera.position,
                    &(vec3){forward.x * velocity, forward.y * velocity, forward.z * velocity});

            if (input_get_key(GLFW_KEY_S))
                camera.position = vec3_subtract(&camera.position,
                    &(vec3){forward.x * velocity, forward.y * velocity, forward.z * velocity});

            if (input_get_key(GLFW_KEY_A))
                camera.position = vec3_subtract(&camera.position,
                    &(vec3){right.x * velocity, right.y * velocity, right.z * velocity});

            if (input_get_key(GLFW_KEY_D))
                camera.position = vec3_add(&camera.position,
                    &(vec3){right.x * velocity, right.y * velocity, right.z * velocity});

            if (input_get_key(GLFW_KEY_Q))
                camera.position = vec3_add(&camera.position, &(vec3){0.0, -velocity, 0.0 });
            
            if (input_get_key(GLFW_KEY_E))
                camera.position = vec3_add(&camera.position, &(vec3){0.0, velocity, 0.0});
        }
        
        vec3 siner = (vec3){0.0, sinf(time_total()) * 0.2, 0.0};
        vec3 total_position = vec3_add(&camera.position, &siner);
        vec3 target = vec3_add(&total_position, &forward);

        view = mat4_look_at(total_position, target, CAMERA_UP);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // cubo girando em Y
        float model[16];
        mat4_identity_raw(model);
        model[0]  =  cosf(cube_angle);
        model[8]  =  sinf(cube_angle);
        model[2]  = -sinf(cube_angle);
        model[10] =  cosf(cube_angle);
        mat4 model_mat = {0}; memcpy(model_mat.m, model, 64);

        renderer3d_draw_mesh(&cube, shader.id, &model_mat, &view, &projection);
        
        
        glUseProgram(shader.id);

        mat4 identity = mat4_identity();

        shader_set_mat4(&shader, "u_model", &identity);
        shader_set_mat4(&shader, "u_view", &view);
        shader_set_mat4(&shader, "u_projection", &projection);

        mesh_draw_lines(&grid);


        renderer3d_begin_batch(&renderer, shader.id, &view, &projection);


        renderer3d_end_batch(&renderer);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mesh_destroy(&cube);
    renderer3d_destroy(&renderer);
    glDeleteTextures(1, &white_tex);
    glfwTerminate();
    return 0;
}