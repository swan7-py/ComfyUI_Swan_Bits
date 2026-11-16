import colorsys
import json
import struct
from io import BytesIO

from comfy_api.latest._util import File3D


def _pad(data, fill):
    remainder = len(data) % 4
    return data if remainder == 0 else data + fill * (4 - remainder)


def _read_glb(raw):
    magic, version, _length = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF":
        raise ValueError("Swan Track Colors expects a binary glTF (.glb) file.")
    if version != 2:
        raise ValueError(f"Unsupported glTF version {version}; only 2 is handled.")
    gltf, bin_chunk, offset = None, b"", 12
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack("<I4s", raw[offset:offset + 8])
        offset += 8
        chunk = raw[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk
    if gltf is None:
        raise ValueError("The .glb has no JSON chunk.")
    return gltf, bin_chunk


def _write_glb(gltf, bin_chunk):
    payload = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    chunks = struct.pack("<I4s", len(payload), b"JSON") + payload
    if bin_chunk:
        body = _pad(bin_chunk, b"\x00")
        chunks += struct.pack("<I4s", len(body), b"BIN\x00") + body
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _to_linear(rgb):
    # glTF baseColorFactor is linear; hex/HSV give sRGB. Same 2.2 curve the
    # SAM3D voxel painter uses for GLTF vertex colours.
    return tuple(c ** 2.2 for c in rgb)


def resolve_colors(spec, count, to_linear=True):
    """Resolve a colour spec into `count` RGB triples.

    GLB materials need linear values (baseColorFactor is linear); the video
    renderer writes float RGB straight into an IMAGE, which is sRGB.
    """
    picked = []
    for token in spec.replace(";", ",").split(","):
        token = token.strip().lstrip("#")
        if not token:
            continue
        if len(token) != 6:
            raise ValueError(f"'{token}' is not a 6-digit hex colour (e.g. e05a4f).")
        picked.append(tuple(int(token[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))
    if not picked:
        # Empty spec: even hue spread so overlapping figures stay readable.
        picked = [colorsys.hsv_to_rgb(i / count, 0.65, 0.9) for i in range(count)]
        return picked if not to_linear else [_to_linear(c) for c in picked]
    colors = [picked[i % len(picked)] for i in range(count)]
    return [_to_linear(c) for c in colors] if to_linear else colors


class SWAN_SAM3DTrackColor:
    """Give every SAM3D track its own PBR material and pull the figures apart.

    SAM3DBody exports each tracked person into its own mesh + skin + track root,
    but every root sits at the origin of that person's own camera frame, so all
    figures overlap. This rewrites the .glb bytes: one material per track, an
    evenly spaced X offset per root, and (by default) drops the baked COLOR_0
    vertex colours so they don't multiply against the new material colour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_3d": ("FILE_3D", {
                    "tooltip": "Animated .glb from Create 3D Animation File (BuildPoseFile).",
                }),
                "spacing": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Gap between figures along X, in metres. 0 keeps them stacked.",
                }),
                "keep_vertex_colors": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep the COLOR_0 attribute. On, it multiplies with the material "
                               "colour and darkens it; only useful if you want the rainbow bake kept.",
                }),
                "colors": ("STRING", {
                    "default": "",
                    "tooltip": "Comma separated hex colours, one per track. Empty = automatic "
                               "even hue spread for any number of people.",
                }),
            }
        }

    RETURN_TYPES = ("FILE_3D",)
    RETURN_NAMES = ("model_3d",)
    FUNCTION = "apply"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Per-track solid PBR colouring + X spacing for SAM3DBody GLBs. "
        "Removes the overlapping-at-origin problem and the rainbow vertex bake."
    )

    def apply(self, model_3d, spacing, keep_vertex_colors, colors=""):
        gltf, bin_chunk = _read_glb(model_3d.get_bytes())

        scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
        roots = [i for i in scene.get("nodes", [])
                 if str(gltf["nodes"][i].get("name", "")).startswith("track")]
        if not roots:
            raise ValueError("No 'trackNN' root nodes found; feed a SAM3DBody .glb.")
        roots.sort(key=lambda i: gltf["nodes"][i]["name"])

        palette = resolve_colors(colors or "", len(roots))
        materials = gltf.setdefault("materials", [])

        for slot, root_idx in enumerate(roots):
            root = gltf["nodes"][root_idx]
            r, g, b = palette[slot]

            materials.append({
                "name": f"{root['name']}_mat",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [r, g, b, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            })
            mat_idx = len(materials) - 1

            # A track root holds the armature and the body mesh node.
            mesh_nodes = [root_idx] + root.get("children", [])
            for node_idx in mesh_nodes:
                mesh_idx = gltf["nodes"][node_idx].get("mesh")
                if mesh_idx is None:
                    continue
                for prim in gltf["meshes"][mesh_idx].get("primitives", []):
                    prim["material"] = mat_idx
                    if not keep_vertex_colors:
                        prim.get("attributes", {}).pop("COLOR_0", None)

            if spacing > 0:
                offset = (slot - (len(roots) - 1) / 2.0) * spacing
                root["translation"] = [offset, 0.0, 0.0]

        # The pre-existing white materials are now unreferenced; drop them and
        # renumber so the file doesn't carry dead entries.
        used = sorted({prim["material"]
                       for mesh in gltf["meshes"]
                       for prim in mesh.get("primitives", [])
                       if prim.get("material") is not None})
        remap = {old: new for new, old in enumerate(used)}
        for mesh in gltf["meshes"]:
            for prim in mesh.get("primitives", []):
                if prim.get("material") is not None:
                    prim["material"] = remap[prim["material"]]
        gltf["materials"] = [materials[i] for i in used]

        return (File3D(BytesIO(_write_glb(gltf, bin_chunk)), file_format="glb"),)
