// NVN2 DrawMeshTasks replay (VK_EXT_mesh_shader + Eden task/mesh SPIR-V).
#pragma once
#include "vk_exec.h"

void vk_mesh_op_geo_mode(VkExec *ex, const void *payload);
void vk_mesh_op_draw(VkExec *ex, const void *payload);
int vk_mesh_available(void);   // 1 when VK_EXT_mesh_shader was enabled
