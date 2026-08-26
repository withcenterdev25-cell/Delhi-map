import bpy
import math
from mathutils import Vector


OUTPUT = "/Users/withcenter10/Desktop/Delhi-India_map/assets/delhi-landmark-buildings-isometric/01-exact-2to1-camera-calibration.png"


def material(name, color, roughness=0.65, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def cube(name, location, scale, mat, bevel=0.025):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (scale[0] / 2, scale[1] / 2, scale[2] / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("Fine edge bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
    obj.data.materials.append(mat)
    return obj


def add_front_window_band(x, width, y, z, glass, trim):
    cube("Front glass band", (x, y - 0.035, z), (width, 0.12, 0.52), glass, 0.01)
    cube("Front sill", (x, y - 0.105, z - 0.36), (width + 0.12, 0.18, 0.16), trim, 0.015)


def add_side_window_band(x, y, depth, z, glass, trim):
    cube("Side glass band", (x + 0.035, y, z), (0.12, depth, 0.52), glass, 0.01)
    cube("Side sill", (x + 0.105, y, z - 0.36), (0.18, depth + 0.12, 0.16), trim, 0.015)


bpy.ops.wm.read_factory_settings(use_empty=True)

stone = material("Warm weathered beige stone", (0.58, 0.42, 0.27), 0.9)
stone_light = material("Pale stone trim", (0.72, 0.57, 0.40), 0.82)
red = material("Reddish vertical fins", (0.48, 0.19, 0.10), 0.88)
glass = material("Dark recessed glazing", (0.025, 0.045, 0.052), 0.24, 0.08)
recess = material("Deep tower openings", (0.018, 0.014, 0.012), 1.0)
roof = material("Aged flat roof", (0.38, 0.31, 0.24), 0.96)

# Architectural massing: a tall central slab with stepped side wings.
cube("Central slab", (0, 0, 7.0), (6.2, 4.4, 14.0), stone)
cube("Left wing", (-4.1, 0.7, 6.0), (2.3, 5.8, 12.0), stone)
cube("Far left wing", (-6.0, 1.5, 5.1), (1.7, 5.0, 10.2), stone)
cube("Right wing", (4.1, 0.7, 6.0), (2.3, 5.8, 12.0), stone)
cube("Far right wing", (6.0, 1.5, 5.1), (1.7, 5.0, 10.2), stone)

# Roof surfaces and shallow parapets keep the elevated view legible.
for x, y, w, d, z in [
    (0, 0, 6.0, 4.2, 14.04),
    (-4.1, 0.7, 2.1, 5.6, 12.04),
    (-6.0, 1.5, 1.5, 4.8, 10.24),
    (4.1, 0.7, 2.1, 5.6, 12.04),
    (6.0, 1.5, 1.5, 4.8, 10.24),
]:
    cube("Roof", (x, y, z), (w, d, 0.12), roof, 0.01)

# Repeated facade bands on the camera-facing front and right sides.
for z in [1.0, 2.15, 3.3, 4.45, 5.6, 6.75, 7.9, 9.05, 10.2, 11.35, 12.5]:
    add_front_window_band(0, 5.15, -2.205, z, glass, stone_light)
for z in [1.0, 2.15, 3.3, 4.45, 5.6, 6.75, 7.9, 9.05, 10.2]:
    add_front_window_band(-4.1, 1.55, -2.205, z, glass, stone_light)
    add_front_window_band(4.1, 1.55, -2.205, z, glass, stone_light)
for z in [1.0, 2.15, 3.3, 4.45, 5.6, 6.75, 7.9, 9.05]:
    add_front_window_band(-6.0, 1.05, -1.005, z, glass, stone_light)
    add_front_window_band(6.0, 1.05, -1.005, z, glass, stone_light)

for z in [1.0, 2.15, 3.3, 4.45, 5.6, 6.75, 7.9, 9.05, 10.2, 11.35, 12.5]:
    add_side_window_band(3.105, 0, 3.35, z, glass, stone_light)
for z in [1.0, 2.15, 3.3, 4.45, 5.6, 6.75, 7.9, 9.05]:
    add_side_window_band(6.855, 1.5, 3.8, z, glass, stone_light)

# Tall red fins, with pale frames and dark square openings at the crown.
fin_specs = [
    (-3.0, -2.38, 15.2), (3.0, -2.38, 15.2),
    (-5.15, -2.38, 13.2), (5.15, -2.38, 13.2),
    (-6.82, -1.12, 11.3), (6.82, -1.12, 11.3),
    (3.34, 2.0, 15.2), (6.94, 3.85, 11.3),
]
for x, y, height in fin_specs:
    cube("Red vertical fin", (x, y, height / 2), (0.54, 0.58, height), red, 0.035)
    cube("Pale fin frame", (x, y - 0.02, height - 0.65), (0.68, 0.68, 1.55), stone_light, 0.025)
    cube("Dark fin crown opening", (x, y - 0.38, height - 0.62), (0.34, 0.08, 0.86), recess, 0.01)

# Ground-contact plinth only; no platform or scenery.
cube("Building plinth", (0, 0.55, 0.10), (14.2, 7.2, 0.20), stone_light, 0.02)

# Exact 2:1 dimetric orthographic camera. At azimuth 45 degrees and elevation
# 30 degrees, the projected ground-axis slope is sin(30)=0.5 => 26.565 degrees.
target = Vector((0, 0.55, 6.8))
azimuth = math.radians(45.0)
elevation = math.radians(30.0)
distance = 38.0
direction = Vector((
    math.cos(elevation) * math.cos(azimuth),
    -math.cos(elevation) * math.sin(azimuth),
    math.sin(elevation),
))

bpy.ops.object.camera_add(location=target + direction * distance)
camera = bpy.context.object
camera.name = "Exact orthographic 2-to-1 camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = 22.0
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = camera

# Neutral studio illumination that does not alter silhouette geometry.
bpy.ops.object.light_add(type="AREA", location=(-8, -10, 24))
key = bpy.context.object
key.data.energy = 1450
key.data.shape = "DISK"
key.data.size = 8
key.rotation_euler = ((target - key.location).to_track_quat("-Z", "Y").to_euler())

bpy.ops.object.light_add(type="AREA", location=(12, 4, 14))
fill = bpy.context.object
fill.data.energy = 650
fill.data.size = 10
fill.rotation_euler = ((target - fill.location).to_track_quat("-Z", "Y").to_euler())

world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("Neutral World")
    bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.055, 0.065, 0.075, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1200
scene.render.resolution_y = 1200
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.render.filepath = OUTPUT
scene.render.image_settings.color_depth = "8"
scene.view_settings.look = "AgX - Medium High Contrast"

bpy.ops.wm.save_as_mainfile(filepath="/Users/withcenter10/Desktop/Delhi-India_map/assets/delhi-landmark-buildings-isometric/01-exact-2to1-camera-calibration.blend")
bpy.ops.render.render(write_still=True)
print(f"Rendered {OUTPUT}")
