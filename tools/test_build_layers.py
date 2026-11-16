"""Run with the same uv --with dependencies as build-layers.py."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "build-layers.py"
spec = importlib.util.spec_from_file_location("build_layers", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildLayersTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        self.rig["image"] = {"width": 100, "height": 80}
        for key in ("hand", "buns", "accessories", "strands"):
            self.rig.pop(key, None)
        self.rig["eyes"] = []
        image = Image.new("RGBA", (100, 80), (190, 210, 230, 255))
        draw = ImageDraw.Draw(image)
        for left in (15, 65):
            opening = [[left, 28], [left + 15, 28], [left + 15, 36], [left, 36]]
            roi = [[left - 4, 20], [left + 19, 20], [left + 19, 40], [left - 4, 40]]
            self.rig["eyes"].append({"opening": opening, "roi": roi})
            draw.rectangle((left, 28, left + 15, 36), fill=(255, 255, 255, 255))
            draw.line((left, 27, left + 15, 27), fill=(15, 25, 50, 255), width=2)
        image.save(self.project / "source.png")
        self.draft = self.project / "rig.draft.json"
        self.draft.write_text(json.dumps(self.rig))

    def run_cli(self, *options):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.project), "--rig", "rig.draft.json", *options],
            capture_output=True,
            text=True,
            check=False,
        )

    def add_skirt(self):
        self.rig["cloth"] = [{
            "name": "skirt", "outline": [[20, 45], [80, 45], [80, 70], [20, 70]],
            "waist": [[30, 45], [70, 45]], "hem": [[20, 70], [80, 70]],
        }]
        self.draft.write_text(json.dumps(self.rig))
        with Image.open(self.project / "source.png") as source:
            full = source.convert("RGBA")
        ImageDraw.Draw(full).rectangle((20, 45, 80, 70), fill=(255, 80, 150, 255))
        full.save(self.project / "source.png")
        return full

    def test_cloth_under_keep_keeps_the_garments_and_never_reads_source_under(self):
        self.add_skirt()
        # An unreadable source_under.png proves the option never opens it.
        (self.project / "source_under.png").write_bytes(b"not a png")
        result = self.run_cli("--cloth-under", "keep")
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(self.project / "built/base.png") as base, Image.open(self.project / "built/base_cloth.png") as bare:
            self.assertEqual(np.array(bare).tobytes(), np.array(base).tobytes())
            self.assertEqual(bare.getpixel((50, 60)), (255, 80, 150, 255))
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertIn("cloth_skirt", meta["layers"])
        # Without the option the same project still reads source_under.png (and fails on this one).
        self.assertNotEqual(self.run_cli().returncode, 0)

    def test_hairmask_replaces_colour_clustering(self):
        mask = Image.new("L", (100, 80), 0)
        ImageDraw.Draw(mask).rectangle((0, 0, 99, 18), fill=255)  # hair across the top
        ImageDraw.Draw(mask).rectangle((10, 20, 40, 45), fill=255)  # and over the left eye
        mask.save(self.project / "hair.png")
        result = self.run_cli("--hairmask", str(self.project / "hair.png"))
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(self.project / "built/hairmask.png") as image:
            hair = np.array(image)
        self.assertGreater(int(hair[5, 50]), 128)  # inside the given hair
        self.assertEqual(int(hair[30, 22]), 0)  # the eye region stays out
        self.assertEqual(int(hair[70, 50]), 0)  # far from it
        plain = self.run_cli()
        self.assertEqual(plain.returncode, 0, plain.stderr)
        with Image.open(self.project / "built/hairmask.png") as image:
            self.assertEqual(image.getextrema(), (0, 0))  # colour clustering finds no hair here

    def test_hairmask_must_match_the_source_size(self):
        (self.project / "rig.json").write_text("existing rig")
        (self.project / "built").mkdir()
        (self.project / "built/layers.json").write_text("existing layers")
        Image.new("L", (99, 80), 255).save(self.project / "hair.png")
        result = self.run_cli("--hairmask", str(self.project / "hair.png"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--hairmask", result.stderr)
        self.assertEqual((self.project / "rig.json").read_text(), "existing rig")
        self.assertEqual((self.project / "built/layers.json").read_text(), "existing layers")

    def test_options_change_the_build_id_only_when_given(self):
        self.add_skirt()

        def build_id(*options):
            result = self.run_cli(*options)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads((self.project / "built/layers.json").read_text())["build"]

        plain = build_id()
        self.assertEqual(build_id("--cloth-under", "auto"), plain)
        self.assertNotEqual(build_id("--cloth-under", "keep"), plain)
        self.assertNotEqual(build_id("--open-eyes"), plain)

    def test_open_eyes_keep_the_source_around_each_eye_with_a_hole_for_the_ball(self):
        plain = self.run_cli()
        self.assertEqual(plain.returncode, 0, plain.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertFalse([name for name in meta["layers"] if name.endswith("_open")])
        self.assertFalse(list((self.project / "built").glob("eye*_open.png")))
        result = self.run_cli("--open-eyes")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        with Image.open(self.project / "source.png") as image:
            source = np.array(image.convert("RGBA"))
        for i, left in enumerate((15, 65)):
            x, y, w, h = meta["layers"][f"eye{i}_open"]
            with Image.open(self.project / f"built/eye{i}_open.png") as image:
                layer = np.array(image.convert("RGBA"))
            self.assertEqual(layer.shape[:2], (h, w))

            def at(px, py):
                return layer[py - y, px - x]

            self.assertEqual(at(left + 7, 27).tolist(), source[27, left + 7].tolist())  # the lash line
            self.assertEqual(int(at(left + 7, 28)[3]), 255)  # the opening's edge stays over the ball's clip
            self.assertEqual(int(at(left + 7, 32)[3]), 0)  # the ball itself shows through
            self.assertEqual(int(at(left - 6, 30)[3]), 255)  # the eye region and its margin
            self.assertTrue(0 < int(at(left - 9, 30)[3]) < 255)  # fading out beyond them
            self.assertTrue(x > left - 13 and x + w <= left + 32)

    def test_cloth_layers_keep_legacy_base_and_generate_bare_base(self):
        self.rig['cloth'] = [{
            'name': 'skirt', 'outline': [[20, 45], [80, 45], [80, 70], [20, 70]],
            'waist': [[30, 45], [70, 45]], 'hem': [[20, 70], [80, 70]],
        }]
        self.draft.write_text(json.dumps(self.rig))
        with Image.open(self.project / 'source.png') as source:
            full = source.convert('RGBA')
        ImageDraw.Draw(full).rectangle((20, 45, 80, 70), fill=(255, 80, 150, 255))
        full.save(self.project / 'source.png')
        for with_under in (False, True):
            with self.subTest(with_under=with_under):
                if with_under:
                    under = full.copy()
                    ImageDraw.Draw(under).rectangle((20, 45, 80, 70), fill=(120, 160, 200, 255))
                    under.save(self.project / 'source_under.png')
                result = self.run_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                meta = json.loads((self.project / 'built/layers.json').read_text())
                self.assertIn('cloth_skirt', meta['layers'])
                with Image.open(self.project / 'built/base.png') as image:
                    self.assertEqual(image.getpixel((50, 60)), (255, 80, 150, 255))
                with Image.open(self.project / 'built/base_cloth.png') as image:
                    self.assertEqual(image.size, full.size)
                    self.assertNotEqual(image.getpixel((50, 60)), (255, 80, 150, 255))
                    if with_under:
                        self.assertEqual(image.getpixel((50, 60)), (120, 160, 200, 255))
                with Image.open(self.project / 'built/cloth_skirt.png') as image:
                    self.assertGreater(image.getchannel('A').getextrema()[1], 0)

    def add_limb(self):
        """A dangling leg over a transparent background: its upper part overlaps the body block."""
        image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 10, 95, 50), fill=(190, 210, 230, 255))  # the body, with the eyes on it
        for left in (15, 65):
            draw.rectangle((left, 28, left + 15, 36), fill=(255, 255, 255, 255))
            draw.line((left, 27, left + 15, 27), fill=(15, 25, 50, 255), width=2)
        draw.rectangle((40, 44, 52, 78), fill=(200, 120, 60, 255))  # the leg, hanging below the body
        draw.line((40, 78, 52, 78), fill=(200, 120, 60, 128))  # an antialiased foot edge
        image.save(self.project / "source.png")
        self.rig["limbs"] = [{
            "name": "leg_left", "kind": "leg", "joints": [[46, 45], [46, 62], [46, 77]],
            "outline": [[38, 43], [54, 43], [54, 79], [38, 79]],
        }]
        self.draft.write_text(json.dumps(self.rig))
        return np.array(image)

    def test_limbs_cut_their_layer_and_a_bare_base_that_continues_only_visible_pixels(self):
        source = self.add_limb()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertEqual(meta["bare"], "base_bare")
        x, y, w, h = meta["layers"]["limb_leg_left"]
        with Image.open(self.project / "built/limb_leg_left.png") as image:
            limb = np.array(image.convert("RGBA"))
        with Image.open(self.project / "built/base_bare.png") as image:
            bare = np.array(image.convert("RGBA"))
        with Image.open(self.project / "built/base.png") as image:
            base = np.array(image.convert("RGBA"))
        self.assertEqual(limb.shape[:2], (h, w))
        # the layer is the source inside the outline, alpha included
        self.assertEqual(limb[60 - y, 46 - x].tolist(), source[60, 46].tolist())
        self.assertEqual(limb[78 - y, 46 - x].tolist(), source[78, 46].tolist())
        # base.png keeps the limb for older runtimes; the bare base continues the body under the limb's top
        self.assertEqual(base[60, 46].tolist(), source[60, 46].tolist())
        self.assertEqual(int(bare[47, 46, 3]), 255)
        self.assertLess(np.abs(bare[47, 46, :3].astype(int) - (190, 210, 230)).max(), 12)
        # ...and stays transparent where the limb hangs over the background, or behind its soft edge
        self.assertEqual(int(bare[65, 46, 3]), 0)
        self.assertEqual(int(bare[78, 46, 3]), 0)
        # outside the limb the bare base is the base
        self.assertEqual(bare[30, 90].tolist(), base[30, 90].tolist())
        # at rest the limb over the bare base shows the source exactly
        over = bare.astype(np.float32)
        layer = np.zeros_like(over)
        layer[y:y + h, x:x + w] = limb
        a = layer[..., 3:4] / 255
        rgb = layer[..., :3] * a + over[..., :3] * over[..., 3:4] / 255 * (1 - a)
        alpha = layer[..., 3:4] + over[..., 3:4] * (1 - a)
        region = (slice(43, 80), slice(38, 55))
        expected = source[..., :3].astype(np.float32) * source[..., 3:4] / 255
        self.assertLess(np.abs(rgb[region] - expected[region]).max(), 1.0)
        self.assertLess(np.abs(alpha[region][..., 0] - source[region][..., 3]).max(), 1.0)
        rig = json.loads((self.project / "rig.json").read_text())
        self.assertEqual(rig["limbs"][0]["name"], "leg_left")

    def test_limb_input_checks(self):
        self.add_limb()
        (self.project / "built").mkdir(exist_ok=True)
        (self.project / "built/layers.json").write_text("existing layers")
        for change, message in (
            ({"name": "leg-left"}, "limbs[0].name"),
            ({"kind": "tail"}, "limbs[0].kind"),
            ({"joints": [[46, 45], [46, 62]]}, "limbs[0].joints"),
            ({"outline": [[38, 43], [54, 43], [54, 99]]}, "limbs[0].outline"),
            ({"outline": [[38, 43], [40, 45], [42, 47]]}, "positive area"),
            ({"swing": 11}, "limbs[0].swing"),
        ):
            with self.subTest(change=change):
                rig = copy.deepcopy(self.rig)
                rig["limbs"][0].update(change)
                self.draft.write_text(json.dumps(rig))
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual((self.project / "built/layers.json").read_text(), "existing layers")

    def test_draft_build_is_loadable_without_optional_parts_or_eye_curves(self):
        before = self.draft.read_bytes()
        sprites = self.project / "built/sprites"
        sprites.mkdir(parents=True)
        (sprites / "keep.txt").write_text("keep")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.draft.read_bytes(), before)
        self.assertEqual((sprites / "keep.txt").read_text(), "keep")
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertEqual(meta["version"], 1)
        self.assertEqual(list(meta), ["size", "build", "layers", "eyes", "version"])
        rig = json.loads((self.project / "rig.json").read_text())
        self.assertEqual(len(meta["layers"]), 8)
        self.assertNotIn("hand", meta["layers"])
        for eye in rig["eyes"]:
            self.assertEqual(len(eye["top"]), 24)
            self.assertEqual(len(eye["bot"]), 24)
            self.assertTrue(all(top <= bot for top, bot in zip(eye["top"], eye["bot"])))
        for name, (x, y, width, height) in meta["layers"].items():
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertTrue(0 <= x < x + width <= 100 and 0 <= y < y + height <= 80)
            with Image.open(self.project / "built" / f"{name}.png") as image:
                self.assertEqual(image.size, (width, height))
        with Image.open(self.project / "built/hairmask.png") as image:
            self.assertEqual(image.getextrema(), (0, 0))
        self.assertEqual(rig["eyes"][0]["x0"], 15)
        self.assertEqual(rig["eyes"][0]["x1"], 30)
        # An absent crease still has a valid transparent texture for the engine.
        with Image.open(self.project / "built/eye0_crease.png") as image:
            crease = np.array(image)
        self.assertEqual(int(crease[..., 3].max()), 0)

    def test_invalid_geometry_does_not_replace_existing_output(self):
        (self.project / "rig.json").write_text("existing rig")
        (self.project / "built").mkdir()
        (self.project / "built/layers.json").write_text("existing layers")
        cases = []
        dimensions = copy.deepcopy(self.rig)
        dimensions["image"]["width"] = 101
        cases.append(dimensions)
        polygon = copy.deepcopy(self.rig)
        polygon["eyes"][0]["opening"] = [[15, 28], [15, 28], [15, 28]]
        cases.append(polygon)
        escaping = copy.deepcopy(self.rig)
        escaping["accessories"] = [{"name": "../escape"}]
        cases.append(escaping)
        for rig in cases:
            with self.subTest(rig=rig):
                self.draft.write_text(json.dumps(rig))
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("build-layers:", result.stderr)
                self.assertEqual(
                    (self.project / "rig.json").read_text(), "existing rig"
                )
                self.assertEqual(
                    (self.project / "built/layers.json").read_text(), "existing layers"
                )

    def test_accessory_boxes_round_outward_clip_and_keep_valid_integers(self):
        for box, expected in [
            ([10.2, 20.8, 30.1, 40.2], [10, 20, 31, 41]),
            ([-5.2, -3.1, 101.7, 81.2], [0, 0, 100, 80]),
            ([10, 20, 30, 40], [10, 20, 30, 40]),
        ]:
            with self.subTest(box=box):
                self.rig["accessories"] = [{"name": "tassel", "box": box, "color": {"redness": 0.5, "minRed": 60}}]
                # Add source pixels that satisfy the accessory's colour thresholds.
                with Image.open(self.project / "source.png") as original:
                    image = original.convert("RGBA")
                ImageDraw.Draw(image).rectangle((22, 21, 25, 25), fill=(220, 30, 30, 255))
                image.save(self.project / "source.png")
                self.draft.write_text(json.dumps(self.rig))
                result = self.run_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                saved = json.loads((self.project / "rig.json").read_text())
                self.assertEqual(saved["accessories"][0]["box"], expected)
                self.assertTrue(all(isinstance(n, int) for n in saved["accessories"][0]["box"]))
                if expected != box:
                    self.assertIn("rounded outward and clipped", result.stdout)
                else:
                    self.assertNotIn("rounded outward and clipped", result.stdout)

    def test_collapsed_or_nonfinite_boxes_fail_without_replacing_output(self):
        (self.project / "rig.json").write_text("existing rig")
        (self.project / "built").mkdir()
        (self.project / "built/layers.json").write_text("existing layers")
        for box in [[101, 20, 110, 40], [20, 80, 30, 90], [20, 20, 20, 40], [30.5, 20, 30.4, 40], [float("nan"), 20, 30, 40]]:
            with self.subTest(box=box):
                self.rig["accessories"] = [{"name": "tassel", "box": box, "color": {"redness": 0.5, "minRed": 60}}]
                self.draft.write_text(json.dumps(self.rig))
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("accessories[0].box", result.stderr)
                self.assertEqual((self.project / "rig.json").read_text(), "existing rig")
                self.assertEqual((self.project / "built/layers.json").read_text(), "existing layers")

    def test_sampled_eye_curves_match_existing_fixture(self):
        rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        for eye in rig["eyes"]:
            sampled = builder.eye_curves(eye)
            for key in ("x0", "x1", "top", "bot"):
                np.testing.assert_array_equal(sampled[key], eye[key])

    def test_hair_reaches_buns_and_sides_without_skin_clothes_or_bright_accessories(
        self,
    ):
        image = Image.new("RGBA", (100, 80), (240, 210, 190, 255))
        draw = ImageDraw.Draw(image)
        hair = (70, 40, 30, 255)
        draw.polygon(
            [
                (15, 5),
                (85, 5),
                (90, 20),
                (85, 50),
                (75, 50),
                (75, 15),
                (25, 15),
                (25, 50),
                (15, 50),
                (10, 20),
            ],
            fill=hair,
        )
        draw.ellipse((-2, 3, 16, 21), fill=hair)
        draw.rectangle((25, 55, 75, 79), fill=(45, 45, 45, 255))
        draw.line((25, 9, 75, 9), fill=(100, 255, 255, 255), width=2)
        draw.rectangle((84, 1, 87, 25), fill=(255, 120, 255, 255))
        rig = copy.deepcopy(self.rig)
        rig["head"] = {"cx": 50, "cy": 30, "rx": 42, "ry": 30}
        rig["buns"] = {"bunL": {"cx": 7, "cy": 12, "rx": 9, "ry": 9}}
        rig["strands"] = [{"nodes": [[40, 6], [60, 6]], "sigma": 3, "max": 3}]
        rig["body"] = {
            "chest": {"cx": 50, "cy": 68, "rx": 20, "ry": 12},
            "shoulders": [],
        }
        for name in ("brow", "mouth", "nose"):
            rig["face"][name] = {"cx": 50, "cy": 45, "rx": 4, "ry": 4}
        mask = builder.hair_mask(np.array(image), rig)
        for y, x in [(12, 7), (45, 20), (45, 80), (6, 60)]:
            self.assertGreater(int(mask[y, x]), 50, (y, x))
        for y, x in [(68, 50), (48, 50), (9, 50), (12, 85)]:
            self.assertEqual(int(mask[y, x]), 0, (y, x))




class HeadLayersTest(unittest.TestCase):
    """--head-plate / --head-footprint (the hair lifted off a clean plate) and --head-layers (told apart by colour)."""

    SKIN = (236, 200, 180, 255)
    HAIR = (52, 36, 30, 255)
    SIZE = (360, 420)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        w, h = self.SIZE
        rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        for key in ("hand", "buns", "accessories", "strands"):
            rig.pop(key, None)
        rig["image"] = {"width": w, "height": h}
        rig["head"].update(cx=180, cy=170, rx=150, ry=170)
        rig["face"] = {
            "nose": {"cx": 180, "cy": 222, "rx": 16, "ry": 16}, "mouth": {"cx": 180, "cy": 262, "rx": 30, "ry": 14},
            "eyeA": {"cx": 130, "cy": 190, "rx": 34, "ry": 24}, "eyeB": {"cx": 230, "cy": 190, "rx": 34, "ry": 24},
            "earR": {"cx": 290, "cy": 200, "rx": 12, "ry": 22}, "earL": {"cx": 70, "cy": 200, "rx": 12, "ry": 22},
            "brow": {"cx": 180, "cy": 160, "rx": 80, "ry": 8, "band": [168, 176]}, "jaw": {"cx": 180, "cy": 290, "rx": 50, "ry": 26, "band": [270, 282]},
        }
        rig["mouth"].update(cx=180, cy=262, angle=0, halfLen=20, bow=2)
        rig["mouth"]["area"] = {"cx": 180, "cy": 264, "rx": 30, "ry": 16, "angle": 0}
        rig["cheeks"] = [[120, 235], [240, 235]]
        rig["mesh"]["fine"] = {"x0": 40, "x1": 320, "y0": 20, "y1": 340, "cell": 7}
        rig["eyes"] = []
        for left in (110, 210):
            rig["eyes"].append({
                "opening": [[left, 182], [left + 40, 182], [left + 40, 198], [left, 198]],
                "roi": [[left - 8, 172], [left + 48, 172], [left + 48, 206], [left - 8, 206]],
            })
        self.rig = rig
        # the plate: a face on a body, no hair in front of the face
        plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(plate)
        draw.ellipse((60, 40, 300, 330), fill=self.SKIN)  # the head
        draw.rectangle((120, 300, 240, 419), fill=(90, 110, 170, 255))  # neck and garment
        for left in (110, 210):
            draw.rectangle((left, 182, left + 40, 198), fill=(250, 250, 250, 255))
            draw.ellipse((left + 12, 182, left + 28, 198), fill=(90, 60, 40, 255))
            draw.line((left, 181, left + 40, 181), fill=(30, 22, 20, 255), width=3)
        draw.line((160, 262, 200, 262), fill=(170, 80, 80, 255), width=3)
        self.plate = plate
        # the source: the same with hair - a cap over the head and a fringe down to the brows, one lock by the left eye
        source = plate.copy()
        draw = ImageDraw.Draw(source)
        draw.pieslice((50, 24, 310, 300), 180, 360, fill=self.HAIR)
        hair = Image.new("L", (w, h), 0)
        mask = ImageDraw.Draw(hair)
        for box in ((80, 100, 280, 166), (62, 150, 96, 250)):
            draw.rectangle(box, fill=self.HAIR)
            mask.rectangle(box, fill=255)
        mask.pieslice((50, 24, 310, 300), 180, 360, fill=255)
        source.save(self.project / "source.png")
        self.plate_path = self.project / "plate.png"
        self.footprint_path = self.project / "footprint.png"
        plate.save(self.plate_path)
        hair.save(self.footprint_path)
        (self.project / "rig.draft.json").write_text(json.dumps(rig))

    def run_cli(self, *options):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.project), "--rig", "rig.draft.json", *options],
            capture_output=True, text=True, check=False,
        )

    def meta(self):
        return json.loads((self.project / "built/layers.json").read_text())

    def layer(self, name):
        """A built layer at the source's size (transparent outside its rectangle)."""
        w, h = self.SIZE
        x, y, lw, lh = self.meta()["layers"][name]
        full = np.zeros((h, w, 4), np.uint8)
        with Image.open(self.project / f"built/{name}.png") as image:
            pixels = np.array(image.convert("RGBA") if image.mode != "L" else image.convert("RGBA"))
        self.assertEqual(pixels.shape[:2], (lh, lw))
        full[y : y + lh, x : x + lw] = pixels
        return full

    @staticmethod
    def over(below, above):
        a = above[..., 3:4].astype(np.float32) / 255
        out = below.astype(np.float32).copy()
        out[..., :3] = below[..., :3] * (1 - a) + above[..., :3] * a
        return out

    def test_the_plate_lifts_the_hair_in_front_of_the_face_and_the_resting_picture_stays_the_source(self):
        plain = self.run_cli("--open-eyes")
        self.assertEqual(plain.returncode, 0, plain.stderr)
        plain_meta = self.meta()
        plain_files = {p.name: p.read_bytes() for p in (self.project / "built").glob("*.png")}
        self.assertFalse({"face_field", "hair_front", "face_skin"} & set(plain_meta["layers"]))

        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        meta = self.meta()
        self.assertNotEqual(meta["build"], plain_meta["build"])
        for name in ("face_field", "hair_front", "face_skin", "eye0_open_under", "eye1_open_under"):
            self.assertIn(name, meta["layers"])
        self.assertLess(meta["head"]["restMean"], 1.0)
        self.assertLessEqual(max(abs(v) for v in meta["head"]["shift"]), 0.2)
        # what older runtimes draw is untouched: the base and every layer they know are the same files
        for name, data in plain_files.items():
            self.assertEqual((self.project / "built" / name).read_bytes(), data, name)

        with Image.open(self.project / "source.png") as image:
            source = np.array(image.convert("RGBA"))
        plate = np.array(self.plate)
        hair, skin, field = self.layer("hair_front"), self.layer("face_skin"), self.layer("face_field")
        # the fringe over the forehead is on the hair layer, entirely, in the source's own colour
        self.assertEqual(hair[130, 180].tolist(), list(self.HAIR))
        self.assertEqual(hair[200, 80].tolist(), list(self.HAIR))  # the lock beside the eye
        # the face below the fringe is not
        self.assertEqual(int(hair[230, 180, 3]), 0)
        self.assertEqual(int(hair[190, 130, 3]), 0)  # an eye
        # under the lifted hair lies the plate: skin, where the base still has the hair
        self.assertEqual(skin[130, 180].tolist(), plate[130, 180].tolist())
        self.assertEqual(int(skin[230, 180, 3]), 0)  # nothing where there was no hair
        # the field: the whole shift on the face, none on the hair far above it
        self.assertEqual(int(field[225, 180, 0]), 255)
        self.assertEqual(int(field[40, 180, 0]), 0)
        # hair far from the moving face stays in the base only (the top of the cap)
        self.assertEqual(int(hair[40, 180, 3]), 0)
        # at rest - the patch over the picture, the hair layer over that - the picture is the source again
        rest = self.over(self.over(source, skin), hair)
        error = np.abs(rest[..., :3] - source[..., :3]).max(-1)[source[..., 3] > 0]
        self.assertLess(float(error.mean()), 0.5)
        self.assertLess(float(np.percentile(error, 99.5)), 16)

    def test_open_eye_layers_of_the_face_without_the_hair_carry_no_hair(self):
        # a strand of the fringe hangs into the right eye's region
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).rectangle((236, 150, 244, 190), fill=self.HAIR)
        source.save(self.project / "source.png")
        with Image.open(self.footprint_path) as image:
            footprint = image.convert("L")
        ImageDraw.Draw(footprint).rectangle((236, 150, 244, 190), fill=255)
        footprint.save(self.footprint_path)
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        plate = np.array(self.plate)
        classic, under, hair = self.layer("eye1_open"), self.layer("eye1_open_under"), self.layer("hair_front")
        # the layer older runtimes draw keeps the strand; the one drawn under the hair layer shows the plate's eye region
        self.assertEqual(classic[178, 240].tolist(), list(self.HAIR))
        self.assertEqual(under[178, 240].tolist(), plate[178, 240].tolist())
        # and the strand itself is on the hair layer, down to its tip
        self.assertEqual(hair[178, 240].tolist(), list(self.HAIR))
        self.assertEqual(hair[188, 240].tolist(), list(self.HAIR))

    def test_a_plate_that_is_no_plate_leaves_the_head_layers_out_and_says_why(self):
        # the "plate" still has the hair: nothing can be lifted
        with Image.open(self.project / "source.png") as image:
            image.save(self.plate_path)
        result = self.run_cli("--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("head layers left out", result.stderr)
        self.assertFalse({"face_field", "hair_front", "face_skin"} & set(self.meta()["layers"]))
        # a plate of another size is an input error: nothing is written
        before = (self.project / "built/layers.json").read_text()
        self.plate.resize((180, 210)).save(self.plate_path)
        result = self.run_cli("--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("head plate", result.stderr)
        self.assertEqual((self.project / "built/layers.json").read_text(), before)
        # the two options go together
        result = self.run_cli("--head-plate", str(self.plate_path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--head-footprint", result.stderr)

    def test_head_layers_by_colour_cut_the_fringe_and_refuse_hair_over_an_eye(self):
        mask = self.project / "hair.png"
        with Image.open(self.footprint_path) as image:
            image.save(mask)
        result = self.run_cli("--hairmask", str(mask), "--head-layers")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        meta = self.meta()
        for name in ("face_field", "hair_front", "face_skin"):
            self.assertIn(name, meta["layers"])
        hair, skin = self.layer("hair_front"), self.layer("face_skin")
        self.assertEqual(hair[130, 180].tolist(), list(self.HAIR))
        self.assertEqual(int(hair[230, 180, 3]), 0)
        # under the fringe: skin continued from the face (no plate to take it from), close to the face's own colour
        self.assertEqual(int(skin[130, 180, 3]), 255)
        self.assertLess(int(np.abs(skin[130, 180, :3].astype(int) - np.array(self.SKIN[:3])).max()), 12)
        # a wide lock over the left eye cannot be told from the eye's own layers: the layers are left out, with the reason
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).rectangle((104, 150, 128, 204), fill=self.HAIR)
        source.save(self.project / "source.png")
        with Image.open(mask) as image:
            wider = image.convert("L")
        ImageDraw.Draw(wider).rectangle((104, 150, 128, 204), fill=255)
        wider.save(mask)
        result = self.run_cli("--hairmask", str(mask), "--head-layers")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("head layers left out: hair hangs over eye 0", result.stderr)
        self.assertFalse({"face_field", "hair_front", "face_skin"} & set(self.meta()["layers"]))


if __name__ == "__main__":
    unittest.main()
