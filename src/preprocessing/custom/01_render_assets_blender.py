"""
Renders preview images of custom assets using Blender headless.

Reads the output structure from organize_custom_assets.py:
  <assets_dir>/
  ├── <uuid-1>/
  │   └── raw_model.glb
  ├── <uuid-2>/
  │   └── raw_model.glb
  └── model_info.json

Produces a 2x2 composite image.jpg per asset (4 camera angles in one image):
  Top-left:     Front 3/4 view
  Top-right:    Back 3/4 view
  Bottom-left:  Side view
  Bottom-right: Top-down view

Usage:
  blender --background --python 01_render_assets_blender.py -- --assets_dir <path>

Optional:
  --resolution 512       Per-tile resolution (final image = 2x resolution, default 512 -> 1024x1024)
  --samples 64           Render samples (default 64)
"""

import bpy
import bmesh
import json
import math
import os
import sys
from mathutils import Vector
from pathlib import Path


def parse_args():
	# Blender passes everything after "--" to the script
	argv = sys.argv
	if "--" in argv:
		argv = argv[argv.index("--") + 1:]
	else:
		argv = []

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--assets_dir", type=str, required=True)
	parser.add_argument("--resolution", type=int, default=512)
	parser.add_argument("--samples", type=int, default=64)
	return parser.parse_args(argv)


def clear_scene():
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)

	for block in bpy.data.meshes:
		if block.users == 0:
			bpy.data.meshes.remove(block)
	for block in bpy.data.materials:
		if block.users == 0:
			bpy.data.materials.remove(block)
	for block in bpy.data.images:
		if block.users == 0:
			bpy.data.images.remove(block)
	for block in bpy.data.cameras:
		if block.users == 0:
			bpy.data.cameras.remove(block)
	for block in bpy.data.lights:
		if block.users == 0:
			bpy.data.lights.remove(block)


def setup_render(resolution, samples):
	scene = bpy.context.scene
	scene.render.engine = 'BLENDER_EEVEE'
	scene.render.resolution_x = resolution
	scene.render.resolution_y = resolution
	scene.render.image_settings.file_format = 'JPEG'
	scene.render.image_settings.quality = 90
	scene.render.film_transparent = True
	scene.eevee.taa_render_samples = samples


def setup_lighting():
	# Key light (warm, from upper-front-right)
	bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
	key = bpy.context.active_object
	key.data.energy = 200
	key.data.size = 4
	key.data.color = (1.0, 0.95, 0.9)
	key.rotation_euler = (math.radians(45), 0, math.radians(45))

	# Fill light (cool, from left)
	bpy.ops.object.light_add(type='AREA', location=(-4, -2, 3))
	fill = bpy.context.active_object
	fill.data.energy = 80
	fill.data.size = 5
	fill.data.color = (0.9, 0.93, 1.0)
	fill.rotation_euler = (math.radians(50), 0, math.radians(-60))

	# Rim light (from behind)
	bpy.ops.object.light_add(type='AREA', location=(0, 4, 4))
	rim = bpy.context.active_object
	rim.data.energy = 120
	rim.data.size = 3
	rim.rotation_euler = (math.radians(-45), 0, 0)

	# Set world background to neutral gray
	world = bpy.data.worlds.get("World")
	if world is None:
		world = bpy.data.worlds.new("World")
	bpy.context.scene.world = world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg:
		bg.inputs[0].default_value = (0.85, 0.85, 0.85, 1.0)
		bg.inputs[1].default_value = 0.5


def get_object_bounds():
	"""Get bounding box center and size across all mesh objects."""
	min_co = Vector((float('inf'),) * 3)
	max_co = Vector((float('-inf'),) * 3)

	for obj in bpy.context.scene.objects:
		if obj.type != 'MESH':
			continue
		bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
		for co in bbox:
			min_co.x = min(min_co.x, co.x)
			min_co.y = min(min_co.y, co.y)
			min_co.z = min(min_co.z, co.z)
			max_co.x = max(max_co.x, co.x)
			max_co.y = max(max_co.y, co.y)
			max_co.z = max(max_co.z, co.z)

	center = (min_co + max_co) / 2
	size = max_co - min_co
	return center, size


def setup_camera_for_view(center, size, azimuth_deg, elevation_deg, fov_deg=40):
	"""Position camera looking at center from given azimuth/elevation."""
	max_dim = max(size.x, size.y, size.z)
	# Distance so object fills frame
	distance = max_dim / (2 * math.tan(math.radians(fov_deg / 2))) * 1.4

	az = math.radians(azimuth_deg)
	el = math.radians(elevation_deg)

	cam_x = center.x + distance * math.cos(el) * math.sin(az)
	cam_y = center.y - distance * math.cos(el) * math.cos(az)
	cam_z = center.z + distance * math.sin(el)

	cam_data = bpy.data.cameras.get("RenderCam")
	if cam_data is None:
		cam_data = bpy.data.cameras.new("RenderCam")
	cam_data.lens_unit = 'FOV'
	cam_data.angle = math.radians(fov_deg)

	cam_obj = bpy.data.objects.get("RenderCamObj")
	if cam_obj is None:
		cam_obj = bpy.data.objects.new("RenderCamObj", cam_data)
		bpy.context.scene.collection.objects.link(cam_obj)

	cam_obj.location = (cam_x, cam_y, cam_z)

	# Point camera at center
	direction = Vector((center.x - cam_x, center.y - cam_y, center.z - cam_z))
	rot_quat = direction.to_track_quat('-Z', 'Y')
	cam_obj.rotation_euler = rot_quat.to_euler()

	bpy.context.scene.camera = cam_obj
	return cam_obj


def render_to_file(filepath):
	bpy.context.scene.render.filepath = filepath
	bpy.ops.render.render(write_still=True)


def composite_images(image_paths, output_path, tile_res):
	"""Combine 4 rendered images into a 2x2 grid using Blender's compositor isn't
	needed - just use the Python imaging approach with the pixels Blender gives us."""
	# Load all 4 images
	imgs = []
	for p in image_paths:
		img = bpy.data.images.load(p)
		imgs.append(img)

	final_w = tile_res * 2
	final_h = tile_res * 2

	# Create output image
	out = bpy.data.images.new("composite", width=final_w, height=final_h)
	pixels = [0.0] * (final_w * final_h * 4)

	# Positions: (col, row) in image space (origin bottom-left)
	# Top-left in display = bottom-left row=tile_res..2*tile_res, col=0..tile_res
	positions = [
		(0, tile_res),       # top-left in display
		(tile_res, tile_res),  # top-right in display
		(0, 0),              # bottom-left in display
		(tile_res, 0),       # bottom-right in display
	]

	for img, (ox, oy) in zip(imgs, positions):
		src = list(img.pixels)
		for y in range(tile_res):
			for x in range(tile_res):
				src_idx = (y * tile_res + x) * 4
				dst_idx = ((oy + y) * final_w + (ox + x)) * 4
				pixels[dst_idx:dst_idx + 4] = src[src_idx:src_idx + 4]

	out.pixels = pixels
	out.filepath_raw = output_path
	out.file_format = 'JPEG'
	out.save_render(output_path)

	# Cleanup
	for img in imgs:
		bpy.data.images.remove(img)
	bpy.data.images.remove(out)


def render_asset(glb_path, output_path, resolution, samples):
	clear_scene()
	setup_render(resolution, samples)
	setup_lighting()

	# Import GLB
	bpy.ops.import_scene.gltf(filepath=str(glb_path))

	center, size = get_object_bounds()

	# 4 camera angles: (azimuth, elevation)
	views = [
		(35, 25),    # front 3/4
		(215, 25),   # back 3/4
		(125, 20),   # side view
		(0, 80),     # top-down
	]

	tmp_dir = str(Path(output_path).parent)
	tmp_paths = []

	for i, (az, el) in enumerate(views):
		setup_camera_for_view(center, size, az, el)
		tmp_path = os.path.join(tmp_dir, f"_tmp_view_{i}.jpg")
		render_to_file(tmp_path)
		tmp_paths.append(tmp_path)

	# Composite into 2x2 grid
	composite_images(tmp_paths, output_path, resolution)

	# Cleanup temp files
	for p in tmp_paths:
		if os.path.exists(p):
			os.remove(p)


def main():
	args = parse_args()
	assets_dir = Path(args.assets_dir)

	model_info_path = assets_dir / "model_info.json"
	assets = json.load(open(model_info_path))
	print(f"Found {len(assets)} assets in {model_info_path}")

	for i, asset in enumerate(assets):
		model_id = asset["model_id"]
		glb_path = assets_dir / model_id / "raw_model.glb"
		output_path = str(assets_dir / model_id / "image.jpg")

		if os.path.exists(output_path):
			print(f"[{i+1}/{len(assets)}] SKIP {model_id} (image.jpg exists)")
			continue

		if not glb_path.exists():
			print(f"[{i+1}/{len(assets)}] SKIP {model_id} (no raw_model.glb)")
			continue

		print(f"[{i+1}/{len(assets)}] Rendering {model_id}...")
		try:
			render_asset(str(glb_path), output_path, args.resolution, args.samples)
		except Exception as e:
			print(f"  FAILED: {e}")

	print(f"\nDone. Rendered images for assets in {assets_dir}")


if __name__ == "__main__":
	main()
