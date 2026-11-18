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

    def test_under_a_limb_the_garment_goes_on_with_its_own_texture(self):
        # an arm hanging in front of a striped jacket: the bare base under it continues the stripes, not a smooth blend
        image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        stripes = [(200, 60, 60, 255), (60, 60, 200, 255)]
        for x in range(5, 96):
            draw.line((x, 10, x, 79), fill=stripes[x // 4 % 2])
        for left in (15, 65):  # the eyes
            draw.rectangle((left, 28, left + 15, 36), fill=(255, 255, 255, 255))
            draw.line((left, 27, left + 15, 27), fill=(15, 25, 50, 255), width=2)
        draw.rectangle((40, 30, 56, 78), fill=(230, 200, 160, 255))  # the arm, over the jacket and below it
        image.save(self.project / "source.png")
        self.rig["limbs"] = [{
            "name": "arm_left", "kind": "arm", "joints": [[48, 32], [48, 55], [48, 77]],
            "outline": [[39, 29], [57, 29], [57, 79], [39, 79]],
        }]
        self.draft.write_text(json.dumps(self.rig))
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(self.project / "built/base_bare.png") as image:
            bare = np.array(image.convert("RGBA")).astype(int)
        under = bare[40:72, 42:55]
        self.assertTrue((under[..., 3] == 255).all())
        near = [np.abs(under[..., :3] - colour[:3]).max(axis=-1) <= 40 for colour in stripes]
        # each pixel is one stripe or the other (a blend of the two is neither), and both go on under the arm
        self.assertGreater((near[0] | near[1]).mean(), 0.8)
        self.assertGreater(near[0].mean(), 0.2)
        self.assertGreater(near[1].mean(), 0.2)

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

    def test_an_accessory_outline_cuts_it_whatever_its_colour_and_a_keyhole_leaves_its_hole(self):
        # a cream bag (no red at all) on a strap loop: the outline is a keyhole polygon round the loop's inside
        outer = [[20, 10], [60, 10], [60, 70], [20, 70]]
        hole = [[30, 20], [30, 40], [50, 40], [50, 20]]
        outline = outer[:1] + hole + hole[:1] + outer
        self.rig["accessories"] = [{"name": "bag", "pivot": [40, 10], "tip": [40, 70], "split": 0.4,
                                    "box": [20, 10, 61, 71], "color": {"redness": 1, "minRed": 255}, "outline": outline}]
        with Image.open(self.project / "source.png") as original:
            image = original.convert("RGBA")
        ImageDraw.Draw(image).rectangle((20, 10, 60, 70), fill=(235, 225, 210, 255))
        image.save(self.project / "source.png")
        rgba = np.array(image)
        mask = builder.accessory_mask(rgba, self.rig["accessories"][0])
        self.assertTrue(mask[60, 40] and mask[15, 25])
        self.assertFalse(mask[30, 40])  # inside the loop
        self.assertFalse(mask[5, 5])
        self.draft.write_text(json.dumps(self.rig))
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        layers = json.loads((self.project / "built/layers.json").read_text())
        self.assertIn("bag", layers["layers"])
        # a hard edge: at rest the bag and what it lies over each draw their own pixels
        with Image.open(self.project / "built/bag.png") as layer:
            self.assertLessEqual(set(np.unique(np.array(layer.convert("RGBA"))[..., 3]).tolist()), {0, 255})
        # behind it the base is filled in from round it, not left with a copy of the bag
        with Image.open(self.project / "built/base.png") as base:
            behind = np.array(base.convert("RGBA"))[15:65, 25:55]
        self.assertFalse(((np.abs(behind[..., :3].astype(int) - (235, 225, 210)).max(axis=-1) < 3) & (behind[..., 3] > 0)).all())
        for bad in ([[20, 10], [60, 10]], [[20, 10], [60, float("nan")], [20, 70]]):
            with self.subTest(outline=bad):
                self.rig["accessories"][0]["outline"] = bad
                self.draft.write_text(json.dumps(self.rig))
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("accessories[0].outline", result.stderr)

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

    def test_a_strand_left_on_the_face_stays_in_the_resting_picture(self):
        # the same strand, not on the footprint: by the eye only hair on the footprint is lifted, so its tip stays on the
        # face - and the plate must not stand in for it under the fringe's rim, or it would vanish from the resting picture
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).rectangle((236, 150, 244, 190), fill=self.HAIR)
        source.save(self.project / "source.png")
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        skin, hair = self.layer("face_skin"), self.layer("hair_front")
        for y in range(169, 174):  # below the fringe, inside the eye's region
            self.assertEqual((int(hair[y, 240, 3]), int(skin[y, 240, 3])), (0, 0), y)
        # hair over the face without it gives back the source everywhere the hair layer reaches
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertLess(meta["head"]["restMean"], 1.0)

    def test_an_eyelid_band_the_plate_draws_a_pixel_off_is_no_hair(self):
        # the fringe reaches down over the right eye's lid, where the source shows the lid's darker band under it and the
        # footprint takes that band for hair; the plate draws the same band two pixels lower
        lid = (150, 100, 90, 255)
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        draw = ImageDraw.Draw(source)
        draw.rectangle((200, 166, 260, 168), fill=self.HAIR)
        draw.rectangle((200, 169, 260, 170), fill=lid)
        source.save(self.project / "source.png")
        with Image.open(self.footprint_path) as image:
            footprint = image.convert("L")
        ImageDraw.Draw(footprint).rectangle((200, 166, 260, 170), fill=255)
        footprint.save(self.footprint_path)
        plate = self.plate.copy()
        ImageDraw.Draw(plate).rectangle((200, 171, 260, 172), fill=lid)
        plate.save(self.plate_path)
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        hair = self.layer("hair_front")
        for x in (215, 230, 245):
            self.assertEqual(int(hair[167, x, 3]), 255, x)  # the fringe is hair down to its edge
            self.assertEqual((int(hair[169, x, 3]), int(hair[170, x, 3])), (0, 0), x)  # the band is the face's
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertLess(meta["head"]["restMean"], 1.0)

    def test_strand_tips_in_a_hole_of_the_footprint_by_the_brow_are_hair_and_the_brow_seen_through_the_hair_is_not(self):
        # the footprint leaves two holes in the fringe by the brow: in one the source has strands (hair the plate's maker
        # left out), in the other the brow, which the plate draws a pixel higher; and the fringe has a small highlight
        # lighter than the plate under it, away from the brow
        brow = (120, 84, 70, 255)
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        draw = ImageDraw.Draw(source)
        draw.rectangle((150, 150, 157, 163), fill=(70, 48, 40, 255))
        draw.rectangle((200, 152, 207, 157), fill=brow)
        draw.rectangle((120, 118, 123, 121), fill=(200, 170, 150, 255))
        source.save(self.project / "source.png")
        with Image.open(self.footprint_path) as image:
            footprint = image.convert("L")
        for box in ((150, 150, 157, 163), (200, 152, 207, 157)):
            ImageDraw.Draw(footprint).rectangle(box, fill=0)
        footprint.save(self.footprint_path)
        plate = self.plate.copy()
        ImageDraw.Draw(plate).rectangle((196, 151, 211, 156), fill=brow)
        plate.save(self.plate_path)
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        hair, skin = self.layer("hair_front"), self.layer("face_skin")
        # the strands stay with the hair, with the plate under them; they do not slide with the face
        self.assertGreater(int(hair[156, 153, 3]), 200)
        self.assertGreater(int(skin[156, 153, 3]), 200)
        # the brow does
        self.assertEqual(int(hair[154, 203, 3]), 0)
        # the highlight is hair: the face does not show through the fringe there once it has turned
        self.assertEqual(int(hair[119, 121, 3]), 255)
        self.assertLess(self.meta()["head"]["restMean"], 1.0)

    def test_an_eyeliner_tail_drawn_over_a_lock_by_the_eye_goes_onto_the_hair_layer(self):
        # a lock right beside the left eye, with the eyeliner's tail drawn across it - in the plate the same tail lies
        # on the skin. Left out of the hair, the tail would leave a notch through which the turning or closing face shows
        tail = (30, 22, 20, 255)
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        draw = ImageDraw.Draw(source)
        draw.rectangle((96, 172, 105, 212), fill=self.HAIR)
        draw.rectangle((93, 185, 109, 187), fill=tail)
        source.save(self.project / "source.png")
        with Image.open(self.footprint_path) as image:
            footprint = image.convert("L")
        ImageDraw.Draw(footprint).rectangle((96, 172, 105, 212), fill=255)
        footprint.save(self.footprint_path)
        plate = self.plate.copy()
        ImageDraw.Draw(plate).rectangle((93, 185, 109, 187), fill=tail)
        plate.save(self.plate_path)
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        hair = self.layer("hair_front")
        for x in (98, 101, 104):
            self.assertEqual(hair[186, x].tolist(), list(tail), x)  # the tail, as drawn, on the lock
        self.assertEqual(int(hair[190, 120, 3]), 0)  # the eye is not hair
        self.assertLess(self.meta()["head"]["restMean"], 1.0)

    def test_a_soft_pixel_no_hair_colour_gives_back_keeps_the_source_below_away_from_the_eyes_too(self):
        # beside the left lock, below the eye's zone, a patch the matte reads as a fifth of hair over the plate's skin but
        # off that line in hue (as skin beside a lock over a plate drawn darker there): no colour a hair can have gives it
        # back, so it is no hair and the source stays below - not a stand-in colour in the resting picture
        patch = (199, 217, 95, 255)
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).rectangle((97, 228, 99, 232), fill=patch)
        source.save(self.project / "source.png")
        result = self.run_cli("--open-eyes", "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        source = np.array(Image.open(self.project / "source.png").convert("RGBA"))
        hair, skin = self.layer("hair_front"), self.layer("face_skin")
        rest = self.over(self.over(source, skin), hair)
        for y in (228, 230, 232):
            for x in (97, 98, 99):
                self.assertEqual(int(hair[y, x, 3]), 0, (x, y))
                self.assertLessEqual(float(np.abs(rest[y, x, :3] - source[y, x, :3]).max()), 1.0, (x, y))
        self.assertEqual(hair[200, 80].tolist(), list(self.HAIR))  # the lock itself is still hair
        self.assertLess(self.meta()["head"]["restMean"], 1.0)

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


class IrisLayersTest(unittest.TestCase):
    """eyes[i].iris: the eye white without its iris and the iris as a sprite (tools/eye/iris.py)."""

    SKIN = (236, 200, 180)
    WHITE = (246, 244, 240)
    RING = (44, 28, 22)
    IRIS = (128, 86, 62)
    PUPIL = (20, 12, 10)
    LASH = (40, 26, 22)
    SIZE = (360, 300)
    CENTRES = ((120, 150), (240, 150))
    HALF = (34, 13)  # the opening: half its width and half its height
    RADIUS = 17  # the iris is taller than the opening, so both lids cut it

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        w, h = self.SIZE
        scale = 4  # drawn four times as large and averaged down, so edges are soft as in a real picture
        yy, xx = (np.mgrid[0 : h * scale, 0 : w * scale].astype(np.float32) + 0.5) / scale
        picture = np.empty((h * scale, w * scale, 3), np.float32)
        picture[:] = self.SKIN
        a, b = self.HALF
        for cx, cy in self.CENTRES:
            reach = b * np.clip(1 - ((xx - cx) / a) ** 2, 0, None)
            inside = (np.abs(xx - cx) < a) & (np.abs(yy - cy) < reach)
            r = np.hypot(xx - cx, yy - cy)
            eye = np.empty_like(picture)
            eye[:] = self.WHITE
            eye[yy < cy - reach + 2.5] = (214, 204, 200)  # the lid's shadow on the eye
            eye[r < self.RADIUS] = self.RING
            eye[r < self.RADIUS - 3] = self.IRIS
            eye[r < 6] = self.PUPIL
            eye[np.hypot(xx - cx - 5, yy - cy + 5) < 2.5] = (255, 255, 255)
            picture[inside] = eye[inside]
            picture[(np.abs(xx - cx) < a + 2) & (yy <= cy - reach) & (yy > cy - reach - 3)] = self.LASH
        picture[(np.abs(xx - 180) < 22) & (np.abs(yy - 232) < 1.5)] = (170, 80, 80)
        small = picture.reshape(h, scale, w, scale, 3).mean((1, 3))
        self.source = np.dstack([np.rint(small), np.full((h, w), 255)]).astype(np.uint8)
        Image.fromarray(self.source).save(self.project / "source.png")
        rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        for key in ("hand", "buns", "accessories", "strands"):
            rig.pop(key, None)
        rig["image"] = {"width": w, "height": h}
        rig["head"].update(cx=180, cy=150, rx=150, ry=140)
        rig["face"] = {
            "nose": {"cx": 180, "cy": 195, "rx": 16, "ry": 16}, "mouth": {"cx": 180, "cy": 232, "rx": 30, "ry": 14},
            "eyeA": {"cx": 120, "cy": 150, "rx": 44, "ry": 26}, "eyeB": {"cx": 240, "cy": 150, "rx": 44, "ry": 26},
            "earR": {"cx": 330, "cy": 160, "rx": 12, "ry": 22}, "earL": {"cx": 30, "cy": 160, "rx": 12, "ry": 22},
            "brow": {"cx": 180, "cy": 110, "rx": 90, "ry": 8, "band": [118, 126]}, "jaw": {"cx": 180, "cy": 262, "rx": 50, "ry": 20, "band": [246, 256]},
        }
        rig["mouth"].update(cx=180, cy=232, angle=0, halfLen=20, bow=0)
        rig["mouth"]["area"] = {"cx": 180, "cy": 234, "rx": 30, "ry": 14, "angle": 0}
        rig["cheeks"] = [[110, 200], [250, 200]]
        rig["mesh"]["fine"] = {"x0": 40, "x1": 320, "y0": 60, "y1": 270, "cell": 7}
        rig["eyes"] = []
        for cx, cy in self.CENTRES:
            xs = np.linspace(cx - a, cx + a, 11)
            arc = b * (1 - ((xs - cx) / a) ** 2)
            opening = [[float(x), float(cy - d)] for x, d in zip(xs, arc)] + [[float(x), float(cy + d)] for x, d in zip(xs[-2:0:-1], arc[-2:0:-1])]
            rig["eyes"].append({
                "opening": opening,
                "roi": [[cx - a - 10, cy - b - 12], [cx + a + 10, cy - b - 12], [cx + a + 10, cy + b + 8], [cx - a - 10, cy + b + 8]],
            })
        self.rig = rig

    def iris(self, eye, **changes):
        cx, cy = self.CENTRES[eye]
        return {"cx": cx, "cy": cy, "rx": self.RADIUS, "ry": self.RADIUS, **changes}

    def build(self, irises, *options):
        rig = copy.deepcopy(self.rig)
        for eye, iris in irises.items():
            rig["eyes"][eye]["iris"] = iris
        (self.project / "rig.draft.json").write_text(json.dumps(rig))
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
            pixels = np.array(image.convert("RGBA"))
        self.assertEqual(pixels.shape[:2], (lh, lw))
        full[y : y + lh, x : x + lw] = pixels
        return full

    def drawn(self, eye, margin=0):
        """The pixels the runtime draws fully from the eye layers at rest."""
        w, h = self.SIZE
        return builder.ball_inside((h, w), self.meta()["eyes"][eye], margin)

    def test_an_eye_with_an_iris_gets_a_white_without_it_and_the_iris_as_a_whole_disc(self):
        result = self.build({0: self.iris(0), 1: self.iris(1)}, "--open-eyes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("iris layers left out", result.stderr)
        meta = self.meta()
        self.assertEqual(len(meta["iris"]), 2)
        luma = lambda rgb: rgb[..., :3].astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
        yy, xx = np.mgrid[0 : self.SIZE[1], 0 : self.SIZE[0]]
        for eye, (cx, cy) in enumerate(self.CENTRES):
            for part in ("ball", "white", "iris", "lash", "low", "crease", "open"):
                self.assertIn(f"eye{eye}_{part}", meta["layers"])
            white, iris, drawn = self.layer(f"eye{eye}_white"), self.layer(f"eye{eye}_iris"), self.drawn(eye)
            r = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
            # nothing of the iris is left in the white: where the iris was it is as light as the white beside it
            inner = self.drawn(eye, 2)
            where = inner & (r < self.RADIUS + 1)
            self.assertTrue((white[..., 3][drawn] == 255).all())
            self.assertGreater(float(luma(white)[where].min()), 185)
            beside = inner & (r > self.RADIUS + 6)
            self.assertLess(abs(float(np.median(luma(white)[where])) - float(np.median(luma(white)[beside]))), 12)
            # ... and along the upper lid it is shaded like the white beside it, not bright up to the lash line
            rim = drawn & ~self.drawn(eye, 1) & (yy < cy)
            self.assertLess(abs(float(np.median(luma(white)[rim & (r < self.RADIUS - 4)])) - float(np.median(luma(white)[rim & (r > self.RADIUS + 6)]))), 30)
            # the white reaches beyond the opening, so the soft lid clip never shows the skin under it
            self.assertEqual(int(white[cy - self.HALF[1] - 2, cx, 3]), 255)
            # the iris is the whole disc: its ring carries on under both lids, and no more than the iris
            for dy in (-(self.RADIUS - 1.5), self.RADIUS - 1.5):
                x, y = int(cx), int(cy + dy)
                self.assertFalse(bool(drawn[y, x]))  # hidden by a lid in the picture
                self.assertGreater(int(iris[y, x, 3]), 200)
                self.assertLess(float(luma(iris)[y, x]), 90)  # the dark ring
            self.assertEqual(int(iris[cy, cx, 3]), 255)
            self.assertLess(float(luma(iris)[cy, cx]), 40)  # the pupil
            self.assertEqual(int(iris[int(cy - self.RADIUS - 5), cx, 3]), 0)
            self.assertEqual(int(iris[cy, int(cx + self.RADIUS + 8), 3]), 0)
            # over each other, with the open-eye layer over both, they are the picture again where the runtime
            # draws them fully: also along the lids, where the lid's share of each pixel went to the open-eye layer
            def over(below, above):
                a = above[..., 3:4].astype(np.float32) / 255
                return below * (1 - a) + above[..., :3] * a

            composed = over(over(white[..., :3].astype(np.float32), iris), self.layer(f"eye{eye}_open"))
            error = np.abs(composed - self.source[..., :3]).max(-1)
            self.assertLessEqual(float(error[inner].max()), 1)
            self.assertLessEqual(float(error[drawn].max()), 24)
            self.assertLess(float(error[drawn].mean()), 0.5)
            self.assertLess(meta["iris"][eye]["rest"]["mean"], 0.5)
            self.assertGreater(meta["iris"][eye]["continued"], 100)
            # the eye ball of older runtimes is as before
            ball = self.layer(f"eye{eye}_ball")
            self.assertTrue((ball[drawn][:, :3] == self.source[drawn][:, :3]).all())

    def test_the_open_eye_layer_of_an_iris_eye_keeps_no_strip_of_the_iris(self):
        result = self.build({0: self.iris(0)}, "--open-eyes")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = self.meta()
        self.assertEqual([entry is None for entry in meta["iris"]], [False, True])
        self.assertFalse({"eye1_white", "eye1_iris"} & set(meta["layers"]))
        with_iris, without = self.layer("eye0_open"), self.layer("eye1_open")
        # the hole is everything the eye layers draw fully. Nothing of the picture stays in it: along the lids
        # only the lid's own share of the pixels it blends into, in the lid's colour, never the iris's
        drawn = self.drawn(0)
        self.assertFalse((with_iris[..., 3][self.drawn(0, 2)] > 0).any())
        lid = drawn & (with_iris[..., 3] > 0)
        self.assertGreater(int(lid.sum()), 8)
        self.assertLess(int(with_iris[..., 3][lid].max()), 255)
        yy = np.mgrid[0 : self.SIZE[1], 0 : self.SIZE[0]][0]
        lower = with_iris[lid & (yy > self.CENTRES[0][1])][:, :3].astype(np.float32)
        self.assertGreater(len(lower), 4)
        self.assertLess(float(np.abs(lower - np.float32(self.SKIN)).max()), 12)  # the lower lid is skin
        upper = with_iris[lid & (yy < self.CENTRES[0][1])][:, :3].astype(np.float32)
        self.assertLess(float(np.abs(upper - np.float32(self.LASH)).max()) if len(upper) else 0.0, 12)  # the upper one the lash
        self.assertTrue((with_iris[..., 3][~drawn & self.drawn(0, -3)] == 255).all())
        # an eye without iris layers keeps its rim, as before
        rim = self.drawn(1) & ~self.drawn(1, 1)
        self.assertTrue((without[..., 3][rim] == 255).all())
        top = self.CENTRES[1][1] - self.HALF[1]
        self.assertEqual(int(without[top + 1, self.CENTRES[1][0], 3]), 255)
        self.assertEqual(int(with_iris[top + 3, self.CENTRES[0][0], 3]), 0)
        self.assertEqual(int(with_iris[top - 2, self.CENTRES[0][0], 3]), 255)  # the lash line stays

    def test_an_eye_that_cannot_be_split_is_cut_as_before_and_says_why(self):
        cases = [
            (self.iris(0, cx=180, cy=232), "the iris outline is not on this eye"),
            (self.iris(0, rx=40, ry=40), "no eye white beside the iris"),
        ]
        plain = self.build({})
        self.assertEqual(plain.returncode, 0, plain.stderr)
        ball = (self.project / "built/eye0_ball.png").read_bytes()
        for iris, reason in cases:
            result = self.build({0: iris})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"iris layers left out: eye 0: {reason}", result.stderr)
            meta = self.meta()
            self.assertNotIn("iris", meta)
            self.assertFalse({"eye0_white", "eye0_iris"} & set(meta["layers"]))
            self.assertEqual((self.project / "built/eye0_ball.png").read_bytes(), ball)

    def test_a_malformed_iris_fails_without_replacing_existing_output(self):
        self.assertEqual(self.build({}).returncode, 0)
        before = (self.project / "built/layers.json").read_bytes()
        for iris in ({"cx": 120, "cy": 150, "rx": 0, "ry": 17}, {"cx": 120, "cy": 150, "rx": 17}, {"cx": -5, "cy": 150, "rx": 17, "ry": 17}, [120, 150, 17, 17]):
            result = self.build({0: iris})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("eyes[0].iris", result.stderr)
            self.assertEqual((self.project / "built/layers.json").read_bytes(), before)


class HemBandTest(unittest.TestCase):
    """--cloth-under hem: the base keeps the garment, except a band along its free edges (tools/cloth/hem.py)."""

    SIZE = (200, 300)
    TORSO = (236, 200, 180, 255)
    SKIRT = (120, 150, 210, 255)
    LINE = (30, 30, 60, 255)
    LEG = (230, 180, 160, 255)
    WAIST, HEM = 110, 220
    LEGS = ((70, 92), (108, 130))  # x ranges; the legs run from under the skirt down to the bottom

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        w, h = self.SIZE
        image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 20, 140, self.WAIST + 4), fill=self.TORSO)  # head and torso
        for x0, x1 in self.LEGS:
            draw.rectangle((x0, 180, x1, h - 1), fill=self.LEG)
        # the skirt, wider at the hem, with a 3 px line around it
        self.skirt = [(58, self.WAIST), (142, self.WAIST), (166, self.HEM), (34, self.HEM)]
        draw.polygon(self.skirt, fill=self.LINE)
        inner = [(61, self.WAIST + 1), (139, self.WAIST + 1), (162, self.HEM - 3), (38, self.HEM - 3)]
        draw.polygon(inner, fill=self.SKIRT)
        image.save(self.project / "source.png")
        self.source = np.array(image)
        rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        for key in ("hand", "buns", "accessories", "strands"):
            rig.pop(key, None)
        rig["image"] = {"width": w, "height": h}
        rig["eyes"] = []
        for left in (72, 112):
            rig["eyes"].append({"opening": [[left, 50], [left + 14, 50], [left + 14, 56], [left, 56]],
                                "roi": [[left - 4, 44], [left + 18, 44], [left + 18, 60], [left - 4, 60]]})
        # the bound outline sits inside the drawn edge (as the pipeline traces it, away from the soft edge)
        self.outline = [[64, self.WAIST + 3], [136, self.WAIST + 3], [158, self.HEM - 5], [42, self.HEM - 5]]
        rig["cloth"] = [{"name": "skirt", "outline": self.outline, "waist": [[64, self.WAIST + 3], [136, self.WAIST + 3]],
                         "hem": [[42, self.HEM - 5], [100, self.HEM - 5], [158, self.HEM - 5]], "max": 12}]
        self.rig = rig
        legs = Image.new("L", (w, h), 0)
        for x0, x1 in self.LEGS:
            ImageDraw.Draw(legs).rectangle((x0, 180, x1, h - 1), fill=255)
        legs.save(self.project / "legs.png")

    def build(self, *options, travel=None):
        rig = copy.deepcopy(self.rig)
        if travel is not None:
            rig["cloth"][0]["max"] = travel
        (self.project / "rig.draft.json").write_text(json.dumps(rig))
        return subprocess.run([sys.executable, str(SCRIPT), str(self.project), "--rig", "rig.draft.json", *options],
                              capture_output=True, text=True, check=False)

    def meta(self):
        return json.loads((self.project / "built/layers.json").read_text())

    def image(self, name):
        with Image.open(self.project / f"built/{name}.png") as picture:
            return np.array(picture.convert("RGBA"))

    def layer(self, name):
        w, h = self.SIZE
        x, y, lw, lh = self.meta()["layers"][name]
        full = np.zeros((h, w, 4), np.uint8)
        full[y : y + lh, x : x + lw] = self.image(name)
        return full

    @staticmethod
    def over(below, above):
        a = above[..., 3:4].astype(np.float32) / 255
        b = below[..., 3:4].astype(np.float32) / 255
        rgb = above[..., :3] * a + below[..., :3] * b * (1 - a)
        alpha = a + b * (1 - a)
        return np.concatenate([rgb / np.maximum(alpha, 1e-6), alpha * 255], -1)

    def test_the_band_follows_the_travel_and_the_picture_stays_the_same_at_rest(self):
        result = self.build("--cloth-under", "hem", "--hem-legs", str(self.project / "legs.png"))
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = self.meta()
        report = meta["clothUnder"]["garments"]["skirt"]
        self.assertEqual(meta["clothUnder"]["mode"], "hem")
        self.assertEqual(report["band"], 1.25 * 12 + 3)
        base, bare, garment = self.image("base"), self.image("base_cloth"), self.layer("cloth_skirt")
        # at rest the garment layer over the base under it is the picture (below the face, whose eyes the base fills)
        rest = self.over(bare.astype(np.float32), garment.astype(np.float32))
        seen = lambda im: im[..., :3] * (im[..., 3:4] / 255)
        error = np.abs(seen(rest) - seen(self.source.astype(np.float32))).max(-1)
        self.assertLessEqual(float(error[self.WAIST - 10 :].max()), 1.0)
        # the band: where base_cloth no longer holds the garment
        changed = np.abs(bare.astype(int) - base.astype(int)).max(-1) > 0
        ys, xs = np.nonzero(changed)
        self.assertGreater(len(ys), 200)
        self.assertGreater(ys.min(), self.WAIST + 3)  # never at or above the waist line
        # it narrows towards the waist: a quarter of the way down it is at most a quarter of the depth at the hem
        quarter = int(self.WAIST + 3 + 0.25 * (self.HEM - self.WAIST))
        row = np.nonzero(changed[quarter])[0]
        self.assertTrue(len(row) <= 2 * (0.25 * report["band"] + 10 + 2))
        # at the sides it is transparent; under the hem it continues the legs, shaded, and is transparent between them
        side = changed.copy()
        side[:, 50:150] = False
        self.assertGreater(int(side.sum()), 50)
        self.assertTrue((bare[..., 3][side] == 0).all())
        y = self.HEM - 8
        for x0, x1 in self.LEGS:
            centre = (x0 + x1) // 2
            self.assertTrue(changed[y, centre])
            self.assertEqual(int(bare[y, centre, 3]), 255)
            ratio = bare[y, centre, :3].astype(float) / np.array(self.LEG[:3])
            self.assertTrue((ratio > 0.6).all() and (ratio <= 1.01).all(), ratio)
        self.assertTrue(changed[y, 100])
        self.assertEqual(int(bare[y, 100, 3]), 0)
        # the garment's drawn edge goes with the garment layer where the band is, not above it
        y = self.HEM - 30
        edge = int(np.nonzero(self.source[y, :, 3] > 0)[0].min())  # the drawn line on the skirt's left side
        self.assertEqual(self.source[y, edge].tolist(), list(self.LINE))
        self.assertEqual(int(garment[y, edge, 3]), 255)
        self.assertEqual(int(bare[y, edge, 3]), 0)
        self.assertEqual(int(garment[self.WAIST + 1, 100, 3]), 0)
        self.assertEqual(bare[self.WAIST + 1, 100].tolist(), base[self.WAIST + 1, 100].tolist())

    def test_along_an_accessory_lying_over_the_skirt_the_band_holds_the_skirt(self):
        # a bag over the skirt's left side near the hem, cut by an outline; the skirt is traced without it
        bag = [[20, 170], [70, 170], [70, 205], [20, 205]]
        image = Image.fromarray(self.source)
        ImageDraw.Draw(image).rectangle((20, 170, 70, 205), fill=(240, 232, 220, 255))
        image.save(self.project / "source.png")
        self.rig["accessories"] = [{"name": "bag", "pivot": [60, 120], "tip": [45, 205], "split": 0.6,
                                    "box": [20, 170, 71, 206], "color": {"redness": 1, "minRed": 255}, "outline": bag}]
        self.rig["cloth"][0]["outline"] = [[64, self.WAIST + 3], [136, self.WAIST + 3], [158, self.HEM - 5], [100, self.HEM - 5],
                                           [72, self.HEM - 5], [72, 168], [52, 168]]
        result = self.build("--cloth-under", "hem", travel=16)
        self.assertEqual(result.returncode, 0, result.stderr)
        bare = self.image("base_cloth")
        # next to the bag (the skirt goes on behind it) the band is the skirt's colour, opaque ...
        self.assertEqual(int(bare[190, 74, 3]), 255)
        self.assertLess(np.abs(bare[190, 74, :3].astype(int) - self.SKIRT[:3]).max(), 40)
        # ... behind the bag, where the skirt went on, the base is opaque, where it hung over the background, transparent
        self.assertEqual(int(bare[200, 66, 3]), 255)
        self.assertEqual(int(bare[175, 24, 3]), 0)
        # the band along the skirt's own hem stays transparent
        self.assertEqual(int(bare[self.HEM - 6, 140, 3]), 0)

    def test_behind_an_accessory_over_a_tiered_skirt_the_tiers_go_on(self):
        # a wide skirt in tiers (bands of two colours) and a bag over its left part, cut by an outline; the skirt is
        # traced without the bag
        image = Image.fromarray(self.source)
        draw = ImageDraw.Draw(image)
        tiers = [(200, 120, 140, 255), (120, 150, 210, 255)]
        for y in range(110, 221):
            draw.line((10, y, 190, y), fill=tiers[(y - 110) // 9 % 2])
        draw.rectangle((14, 150, 64, 200), fill=(240, 232, 220, 255), outline=(60, 40, 30, 255), width=2)
        image.save(self.project / "source.png")
        self.rig["accessories"] = [{"name": "bag", "pivot": [40, 120], "tip": [40, 200], "split": 0.4,
                                    "box": [14, 150, 65, 201], "color": {"redness": 1, "minRed": 255},
                                    "outline": [[14, 150], [64, 150], [64, 200], [14, 200]]}]
        self.rig["cloth"][0]["outline"] = [[12, 112], [188, 112], [188, 217], [12, 217], [12, 203], [67, 203],
                                           [67, 147], [12, 147]]
        self.rig["cloth"][0]["waist"] = [[12, 112], [188, 112]]
        self.rig["cloth"][0]["hem"] = [[12, 217], [100, 217], [188, 217]]
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        base = self.image("base")
        behind = base[156:195, 20:59]
        # the skirt goes on behind the bag, opaque, and nothing of the bag stays
        self.assertTrue((behind[..., 3] == 255).all())
        self.assertFalse((np.abs(behind[..., :3].astype(int) - (240, 232, 220)).max(axis=-1) < 12).any())
        # row by row it is the tier of that row, as beside the bag: the tiers go on, not one smooth colour
        rows = np.median(behind[..., :3].astype(int), axis=1)
        beside = np.median(base[156:195, 100:180, :3].astype(int), axis=1)
        self.assertGreater(float((np.abs(rows - beside).max(axis=1) < 30).mean()), 0.85)
        self.assertGreater(float(rows.std(axis=0).max()), 25)

    def test_without_legs_the_band_is_transparent_and_it_is_as_deep_as_the_travel_allows(self):
        counts = {}
        for travel in (8, 16):
            result = self.build("--cloth-under", "hem", travel=travel)
            self.assertEqual(result.returncode, 0, result.stderr)
            base, bare = self.image("base"), self.image("base_cloth")
            changed = np.abs(bare.astype(int) - base.astype(int)).max(-1) > 0
            self.assertTrue((bare[..., 3][changed] == 0).all())
            self.assertEqual(self.meta()["clothUnder"]["garments"]["skirt"]["legs"], 0)
            # how deep it reaches at the hem, in the middle of the skirt
            column = np.nonzero(changed[:, 100])[0]
            counts[travel] = int(column.max() - column.min() + 1)
            self.assertLessEqual(counts[travel], 1.25 * travel + 3 + 10 + 1)
        self.assertGreater(counts[16], counts[8] + 5)

    def test_hem_never_reads_the_under_drawing_and_needs_its_own_option_for_legs(self):
        (self.project / "source_under.png").write_bytes(b"not a picture")
        keep = self.build("--cloth-under", "keep")
        self.assertEqual(keep.returncode, 0, keep.stderr)
        kept = self.meta()["build"]
        self.assertNotIn("clothUnder", self.meta())
        self.assertEqual(self.image("base_cloth").tolist(), self.image("base").tolist())
        hem = self.build("--cloth-under", "hem")
        self.assertEqual(hem.returncode, 0, hem.stderr)
        self.assertNotEqual(self.meta()["build"], kept)
        wrong = self.build("--hem-legs", str(self.project / "legs.png"))
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("--hem-legs goes with --cloth-under hem", wrong.stderr)
        small = self.project / "small.png"
        Image.new("L", (10, 10), 255).save(small)
        bad = self.build("--cloth-under", "hem", "--hem-legs", str(small))
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--hem-legs small.png is 10 x 10", bad.stderr)


class CheekLayersTest(unittest.TestCase):
    """--cheek-pull: the skin of each cheek from the face's outer contour inwards, which newer runtimes pull past it."""

    SKIN, HAIR, SIZE = HeadLayersTest.SKIN, HeadLayersTest.HAIR, HeadLayersTest.SIZE
    run_cli, meta, layer = HeadLayersTest.run_cli, HeadLayersTest.meta, HeadLayersTest.layer
    over = staticmethod(HeadLayersTest.over)

    def setUp(self):
        HeadLayersTest.setUp(self)
        w, h = self.SIZE
        # hair hanging beside the face and behind it (in the plate and the source alike), below the cap
        behind = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(behind).ellipse((24, 30, 336, 380), fill=self.HAIR)
        ImageDraw.Draw(behind).rectangle((0, 0, w, 140), fill=(0, 0, 0, 0))
        for path in (self.project / "source.png", self.plate_path):
            with Image.open(path) as image:
                Image.alpha_composite(behind, image.convert("RGBA")).save(path)
        with Image.open(self.plate_path) as image:
            self.plate = image.convert("RGBA")
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((24, 30, 336, 380), fill=255)
        draw.rectangle((0, 0, w, 140), fill=0)
        draw.ellipse((60, 40, 300, 330), fill=0)  # the face is in front of that hair
        draw.pieslice((50, 24, 310, 300), 180, 360, fill=255)
        for box in ((80, 100, 280, 166), (62, 150, 96, 250)):
            draw.rectangle(box, fill=255)
        self.hairmask_path = self.project / "hair.png"
        mask.save(self.hairmask_path)

    @staticmethod
    def contour(y):
        """x of the face's left and right contour at row y (the head ellipse)."""
        t = (1 - ((y - 185) / 145) ** 2) ** 0.5
        return 180 - 120 * t, 180 + 120 * t

    def test_each_cheek_follows_the_contour_keeps_off_the_features_and_is_the_base_at_rest(self):
        plain = self.run_cli("--hairmask", str(self.hairmask_path))
        self.assertEqual(plain.returncode, 0, plain.stderr)
        plain_meta = self.meta()
        self.assertNotIn("cheeks", plain_meta)
        self.assertFalse({"cheek_left", "cheek_right"} & set(plain_meta["layers"]))
        result = self.run_cli("--hairmask", str(self.hairmask_path), "--cheek-pull")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("left out", result.stderr)
        meta = self.meta()
        self.assertNotEqual(meta["build"], plain_meta["build"])
        with Image.open(self.project / "built/base.png") as image:
            base = np.array(image.convert("RGBA"))
        for side, index in (("left", 0), ("right", 1)):
            data = meta["cheeks"][side]
            self.assertGreaterEqual(len(data["edge"]), 3)
            self.assertEqual(len(data["reach"]), len(data["edge"]))
            self.assertGreater(data["travel"], 5)
            ys = [y for _, y in data["edge"]]
            self.assertEqual(ys, sorted(ys))  # top to bottom
            self.assertGreater(ys[0], 200)  # below the eyes
            self.assertLess(ys[-1], 313)  # above the chin
            for (x, y), reach in zip(data["edge"], data["reach"]):
                if side == "right" or y > 264:  # above that the lock by the left eye hangs over the contour
                    self.assertLess(abs(x - self.contour(y)[index]), 4, (side, x, y))
                inner = x + reach if side == "left" else x - reach
                for left in (110, 210):  # the reach stops short of the eyes
                    self.assertFalse(left - 8 <= inner <= left + 48 and 172 <= y <= 206, (side, inner, y))
                self.assertGreater(((inner - 180) / 30) ** 2 + ((y - 264) / 16) ** 2, 1)  # and the mouth
            layer = self.layer(f"cheek_{side}")
            for x, y in ((130, 190), (230, 190), (180, 222), (180, 262)):  # eyes, nose, mouth
                self.assertEqual(int(layer[y, x, 3]), 0)
            # it carries the contour (and the line just outside it) and, over the base, is the base
            x0, _ = self.contour(285)
            if side == "left":
                self.assertEqual(int(layer[285, int(x0) + 2, 3]), 255)
            opaque = layer[..., 3] > 0
            self.assertGreater(int(opaque.sum()), 500)
            rest = self.over(base, layer)
            self.assertLessEqual(float(np.abs(rest[..., :3] - base[..., :3]).max()), 1)

    def test_hair_over_a_cheek_shortens_its_pull_or_leaves_it_out(self):
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).rectangle((262, 210, 268, 300), fill=self.HAIR)  # a lock over the right cheek
        source.save(self.project / "source.png")
        with Image.open(self.hairmask_path) as image:
            mask = image.copy()
        ImageDraw.Draw(mask).rectangle((262, 210, 268, 300), fill=255)
        mask.save(self.hairmask_path)
        result = self.run_cli("--hairmask", str(self.hairmask_path), "--cheek-pull")
        self.assertEqual(result.returncode, 0, result.stderr)
        cheeks = self.meta()["cheeks"]
        self.assertIn("left", cheeks)
        if "right" in cheeks:
            self.assertLess(cheeks["right"]["travel"], 0.8 * cheeks["left"]["travel"])
        else:
            self.assertIn("cheek layers left out: right: hair lies over", result.stderr)

    def test_with_a_plate_the_contour_line_moves_from_the_hair_to_the_skin_and_the_picture_stays(self):
        options = ("--hairmask", str(self.hairmask_path), "--head-plate", str(self.plate_path), "--head-footprint", str(self.footprint_path))
        # the plate's contour is drawn with a line that the source does not have, so the matte lifts it as hair
        draw = ImageDraw.Draw(self.plate)
        draw.arc((60, 40, 300, 330), 100, 260, fill=(120, 70, 60, 255), width=2)
        self.plate.save(self.plate_path)
        with Image.open(self.project / "source.png") as image:
            source = image.convert("RGBA")
        ImageDraw.Draw(source).arc((60, 40, 300, 330), 100, 260, fill=(120, 70, 60, 255), width=2)
        source.save(self.project / "source.png")
        plain = self.run_cli(*options)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        before = self.over(self.over(np.zeros((self.SIZE[1], self.SIZE[0], 4), np.uint8) + 255, self.layer("face_skin")), self.layer("hair_front"))
        result = self.run_cli(*options, "--cheek-pull")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = self.meta()
        self.assertIn("left", meta["cheeks"])
        hair, skin = self.layer("hair_front"), self.layer("face_skin")
        after = self.over(self.over(np.zeros_like(hair) + 255, skin), hair)
        # the same picture over anything
        self.assertLessEqual(float(np.abs(after[..., :3] - before[..., :3]).max()), 2)
        # the fringe and the lock stay hair in front; the hair layer keeps nothing on the cheeks' contour
        self.assertEqual(int(hair[130, 180, 3]), 255)
        self.assertEqual(int(hair[200, 80, 3]), 255)
        for y in (240, 270):
            x0, x1 = self.contour(y)
            self.assertEqual(int(hair[y, int(round(x0)) - 1 : int(round(x0)) + 2, 3].max()), 0)
            self.assertEqual(int(hair[y, int(round(x1)) - 1 : int(round(x1)) + 2, 3].max()), 0)

    def test_the_contour_line_moves_to_the_skin_also_where_it_touches_a_lock(self):
        # hair_front has the right contour's drawn line (2 px, a little outside the skin's edge) and a lock crossing the
        # contour at its top: the lock stays in front, the line below it does not, all the way to where it touches
        cheeks = builder.cheek_pull()
        w, h = self.SIZE
        with Image.open(self.project / "source.png") as image:
            drawn = np.array(image.convert("RGBA"))
        with Image.open(self.hairmask_path) as image:
            hairmask = np.array(image.convert("L"))
        hair_front = np.zeros((h, w, 4), np.uint8)
        for y in range(200, 300):
            x = self.contour(y)[1]
            hair_front[y, int(round(x)) + 2 : int(round(x)) + 4] = (120, 70, 60, 255)
        hair_front[216:225, int(round(self.contour(220)[1])) - 12 : 345] = self.HAIR
        head = {"hair": hair_front, "skin": np.zeros((h, w, 4), np.uint8)}
        layers, data, notes, head = cheeks.cut(drawn, hairmask, self.rig, np.zeros((h, w), bool), builder.polygon_mask, head)
        self.assertIn("right", data, notes)
        self.assertEqual(int(head["hair"][220, 330, 3]), 255)  # the lock
        for y in (228, 234, 260):
            x = int(round(self.contour(y)[1]))
            self.assertEqual(int(head["hair"][y, x : x + 6, 3].max()), 0, y)  # the line is the skin's
            self.assertEqual(int(head["skin"][y, x : x + 6, 3].max()), 255, y)

    def cut_with_hair_front(self, hair_front):
        """cheeks.cut on the source with the given hair_front (and nothing in face_skin)."""
        w, h = self.SIZE
        with Image.open(self.project / "source.png") as image:
            drawn = np.array(image.convert("RGBA"))
        with Image.open(self.hairmask_path) as image:
            hairmask = np.array(image.convert("L"))
        head = {"hair": hair_front, "skin": np.zeros((h, w, 4), np.uint8)}
        return builder.cheek_pull().cut(drawn, hairmask, self.rig, np.zeros((h, w), bool), builder.polygon_mask, head)

    def test_the_pull_stays_short_of_hair_in_front_beside_the_contour(self):
        # hair_front has hair beside the lower right contour, outside the face (the dark gap between a jaw and the hair
        # behind it that a plate's matting lifts as hair in front): a cheek pulled into it would pass under it and come
        # out in a notch where it ends, so the pull holds back there and eases back in above it
        cheeks = builder.cheek_pull()
        w, h = self.SIZE
        hair_front = np.zeros((h, w, 4), np.uint8)
        for y in range(276, 330):
            x = int(round(self.contour(y)[1]))
            hair_front[y, x + 2 : x + 26] = self.HAIR
        layers, data, notes, head = self.cut_with_hair_front(hair_front.copy())
        self.assertIn("right", data, notes)
        right = data["right"]
        edge, free = np.array(right["edge"]), np.array(right["free"])
        self.assertEqual(int(head["hair"][290, int(round(self.contour(290)[1])) + 12, 3]), 255)  # it stays in front
        self.assertEqual(float(free[edge[:, 1] < 236].min()), 1.0)  # whole above it
        # beside it the contour does not move out, and above it the contour moved as far as the pull can take it there
        # (wherever it is centred) stays, with the layer's stroke, short of it
        for (x, y), share in zip(edge, free):
            if y >= 272:
                self.assertLessEqual(share, 0.01, y)
        stations = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(edge, axis=0).T))])
        moves = right["travel"] * cheeks.peak(stations / stations[-1]) * free
        steps = np.gradient(edge, axis=0)
        normals = np.stack([steps[:, 1], -steps[:, 0]], 1) / np.hypot(*steps.T)[:, None]  # outward on the right
        for point, normal, move in zip(edge, normals, moves):
            if move > 0.5:
                x, y = np.rint(point + normal * (move + cheeks.STROKE)).astype(int)
                self.assertEqual(int(hair_front[y, x, 3]), 0, (point, move))
        # and it eases in over about a travel of contour (here some six points): no jump between neighbouring points
        self.assertLess(float(np.abs(np.diff(free)).max()), 0.35)
        self.assertGreaterEqual(int(((free > 0.02) & (free < 0.98)).sum()), 3)
        # the left side, with nothing beside it, is as before
        plain = self.cut_with_hair_front(np.zeros((h, w, 4), np.uint8))[1]
        self.assertEqual(data["left"], plain["left"])
        self.assertNotIn("free", plain["right"])

    def test_residue_of_the_matting_beside_the_contour_moves_to_the_skin(self):
        # specks and a faint thin line in hair_front a little outside the right contour, past the contour's own line:
        # residue of the matting, which a pulled cheek would pass under (a dark dot, a short line on the moved cheek).
        # They move to face_skin, which the pulled cheek covers; a thick lock beside the contour stays hair in front
        w, h = self.SIZE
        hair_front = np.zeros((h, w, 4), np.uint8)
        specks = [(240, 14), (241, 14), (250, 16)]
        for y, dx in specks:
            hair_front[y, int(round(self.contour(y)[1])) + dx] = (60, 40, 35, 150)
        for y in range(226, 292):  # longer than a speck, but one pixel wide and beside the contour all along
            hair_front[y, int(round(self.contour(y)[1])) + 12] = (70, 50, 40, 90)
        hair_front[296:306, 280:330] = self.HAIR  # a lock beside the end of the contour, reaching well beyond it
        before = self.over(np.zeros((h, w, 4), np.uint8) + 255, hair_front)
        layers, data, notes, head = self.cut_with_hair_front(hair_front)
        self.assertIn("right", data, notes)
        for y, dx in [*specks, (230, 12), (260, 12), (288, 12)]:
            x = int(round(self.contour(y)[1])) + dx
            self.assertEqual(int(head["hair"][y, x, 3]), 0, (y, dx))
            self.assertGreater(int(head["skin"][y, x, 3]), 0, (y, dx))
        self.assertEqual(int(head["hair"][300, 320, 3]), 255)
        after = self.over(self.over(np.zeros((h, w, 4), np.uint8) + 255, head["skin"]), head["hair"])
        self.assertLessEqual(float(np.abs(after[..., :3] - before[..., :3]).max()), 2)  # the same picture

    def test_thin_hair_across_the_cheek_does_not_hide_the_contour(self):
        # two lock tips, 2 px wide, lying across the right cheek down to its contour: the face is seen beyond them, so they
        # do not hide the contour. The layer reaches the contour along them and the pull is whole there (taken as hiding
        # it, the layer was cut along the tips and its outline stepped out where they end: a shelf with a light corner)
        w, h = self.SIZE
        hair_front = np.zeros((h, w, 4), np.uint8)
        for i in range(46):
            for dx in (0, 9):
                hair_front[226 + i, 248 + i + dx - 9 : 248 + i + dx - 7] = self.HAIR
        layers, data, notes, head = self.cut_with_hair_front(hair_front)
        self.assertIn("right", data, notes)
        self.assertNotIn("free", data["right"])
        image, rect = layers["cheek_right"]
        for y in (236, 246, 256, 266):
            x = int(round(self.contour(y)[1])) - 2
            self.assertEqual(int(image[y - rect[1], x - rect[0], 3]), 255, y)
        self.assertEqual(int(head["hair"][250, 263, 3]), 255)  # the tips stay in front


class StrandLayersTest(unittest.TestCase):
    """--strand-layers (tools/hair/strands.py): a held lock on its own layer over the base filled in behind it."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.rig = json.loads((ROOT / "samples/miko-qipao/rig.json").read_text())
        self.rig["image"] = {"width": 120, "height": 160}
        for key in ("hand", "buns", "accessories", "cloth"):
            self.rig.pop(key, None)
        self.rig["eyes"] = []
        # skin over the lower part, a dark lock hanging over it from x 50..62 between y 40 and 150
        image = Image.new("RGBA", (120, 160), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 0, 110, 159), fill=(236, 200, 180, 255))
        draw.rectangle((50, 40, 62, 150), fill=(40, 30, 28, 255))
        for left in (20, 85):
            self.rig["eyes"].append({"opening": [[left, 12], [left + 15, 12], [left + 15, 18], [left, 18]],
                                     "roi": [[left - 4, 6], [left + 19, 6], [left + 19, 24], [left - 4, 24]]})
            draw.rectangle((left, 12, left + 15, 18), fill=(255, 255, 255, 255))
            draw.line((left, 11, left + 15, 11), fill=(15, 25, 50, 255), width=2)
        image.save(self.project / "source.png")
        hair = Image.new("L", (120, 160), 0)
        ImageDraw.Draw(hair).rectangle((50, 40, 62, 150), fill=255)
        hair.save(self.project / "hair.png")
        self.rig["strands"] = [{"name": "lock", "nodes": [[56, 40], [56, 77], [56, 114], [56, 150]], "sigma": 6, "k": 1, "max": 8}]

    def build(self, *options, reach=24):
        rig = copy.deepcopy(self.rig)
        if reach is not None:
            rig["strands"][0]["reach"] = reach
        (self.project / "rig.draft.json").write_text(json.dumps(rig))
        return subprocess.run([sys.executable, str(SCRIPT), str(self.project), "--rig", "rig.draft.json",
                               "--hairmask", str(self.project / "hair.png"), *options], capture_output=True, text=True, check=False)

    def layer(self, meta, name):
        x, y, w, h = meta["layers"][name]
        full = np.zeros((160, 120, 4), np.uint8)
        with Image.open(self.project / "built" / f"{name}.png") as image:
            full[y : y + h, x : x + w] = np.array(image.convert("RGBA"))
        return full, (x, y, w, h)

    def test_lock_and_filled_plate_draw_the_base_at_rest(self):
        result = self.build("--strand-layers")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertEqual(meta["strands"]["lock"]["lock"], "strand_0")
        self.assertEqual(meta["strands"]["lock"]["under"], "strand_0_under")
        with Image.open(self.project / "built/base.png") as image:
            base = np.array(image.convert("RGBA")).astype(np.float32)
        lock, _ = self.layer(meta, "strand_0")
        plate, (x, y, w, h) = self.layer(meta, "strand_0_under")
        # the plate takes the place of the base in its rect, the lock goes over it: the base again
        a = lock[..., 3:4].astype(np.float32) / 255
        drawn = lock[..., :3] * a + plate[..., :3] * (1 - a)
        window = (slice(y, y + h), slice(x, x + w))
        self.assertLessEqual(np.abs(drawn[window] - base[window][..., :3]).max(), 5)  # TOLERANCE where told apart
        # the lock is the dark hair; under it the plate is skin, opaque (below its root, where it keeps the hair above)
        self.assertTrue((lock[60:140, 52:61, 3] == 255).all())
        self.assertTrue((plate[60:140, 52:61, 3] == 255).all())
        self.assertGreater(plate[70:140, 52:61, 0].min(), 200)
        # the plate reaches as far round the strand as the base bends with it
        self.assertLessEqual(x, 56 - 2 * 6)
        self.assertGreaterEqual(x + w, 56 + 2 * 6)
        self.assertGreaterEqual(meta["strands"]["lock"]["unsure"], 0)

    @staticmethod
    def shrunk(image, k=4):
        """Premultiplied, averaged over k x k blocks: a display's mip level of the texture."""
        image = image.astype(np.float32)
        image = np.concatenate([image[..., :3] * image[..., 3:4] / 255, image[..., 3:4]], axis=2)
        h, w = image.shape[0] // k * k, image.shape[1] // k * k
        return image[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))

    def test_a_lock_cut_out_of_more_hair_shows_no_seam_however_it_is_filtered(self):
        # the hair goes on above the lock's root (and the mask with it): the cut there lies over the hair the plate
        # keeps, so a display's mip levels of the two show the hair, not a light line of the fill under the cut
        image = Image.open(self.project / "source.png").convert("RGBA")
        ImageDraw.Draw(image).rectangle((44, 20, 68, 150), fill=(40, 30, 28, 255))
        for k in range(46, 68, 4):  # strands, so the cut is not across flat colour
            ImageDraw.Draw(image).line((k, 20, k, 150), fill=(70, 52, 44, 255))
        image.save(self.project / "source.png")
        hair = Image.new("L", (120, 160), 0)
        ImageDraw.Draw(hair).rectangle((44, 20, 68, 150), fill=255)
        hair.save(self.project / "hair.png")
        result = self.build("--strand-layers")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        with Image.open(self.project / "built/base.png") as built:
            base = np.array(built.convert("RGBA")).astype(np.float32)
        lock, _ = self.layer(meta, "strand_0")
        plate, (x, y, w, h) = self.layer(meta, "strand_0_under")
        under = base.copy()
        under[y : y + h, x : x + w] = plate[y : y + h, x : x + w]  # the plate replaces the base in its rect
        for k in (1, 2, 4):  # each texture filtered on its own, then the lock drawn over the plate, across the root
            top, bottom = self.shrunk(lock, k), self.shrunk(under, k)
            drawn = top + bottom * (1 - top[..., 3:4] / 255)
            rows, cols = slice(24 // k, 72 // k), slice(48 // k, 64 // k)
            self.assertLessEqual(np.abs(drawn - self.shrunk(base, k))[rows, cols].max(), 6, k)
        # the plate keeps the hair under the root and goes over to the skin further down
        self.assertLess(plate[45:50, 52:61, 0].max(), 100)
        self.assertGreater(plate[100:140, 52:61, 0].min(), 200)

    def test_thin_hairs_go_with_the_lock_and_a_garment_line_under_it_stays(self):
        image = Image.open(self.project / "source.png").convert("RGBA")
        draw = ImageDraw.Draw(image)
        draw.line((20, 100, 100, 100), fill=(120, 60, 50, 255), width=2)  # a seam under the lock, on beyond it
        draw.rectangle((50, 40, 62, 150), fill=(40, 30, 28, 255))
        draw.line((64, 118, 64, 140), fill=(60, 45, 40, 255), width=1)  # a thin hair by it, not in the mask
        image.save(self.project / "source.png")
        result = self.build("--strand-layers")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.project / "built/layers.json").read_text())
        lock, _ = self.layer(meta, "strand_0")
        plate, _ = self.layer(meta, "strand_0_under")
        hair = [(64, 118 + t) for t in range(0, 23)]
        self.assertGreater(min(lock[yy, xx, 3] for xx, yy in hair), 128)
        self.assertGreater(min(plate[yy, xx, 0] for xx, yy in hair), 200)  # skin under it, no hair left behind
        self.assertTrue((lock[100:102, 67:72, 3] == 0).all())  # the seam beside the lock stays in the plate
        self.assertTrue((plate[100:102, 67:72, 0] < 160).all())

    def test_without_reach_or_option_nothing_changes(self):
        self.assertEqual(self.build(reach=None).returncode, 0)
        plain = json.loads((self.project / "built/layers.json").read_text())
        self.assertEqual(self.build("--strand-layers", reach=None).returncode, 0)
        meta = json.loads((self.project / "built/layers.json").read_text())
        self.assertNotIn("strands", meta)
        self.assertFalse(any(name.startswith("strand_") for name in meta["layers"]))
        self.assertEqual(self.build().returncode, 0)  # a reach without the option: no layers either
        self.assertNotIn("strands", json.loads((self.project / "built/layers.json").read_text()))
        self.assertNotEqual(plain["build"], meta["build"])  # the option takes part in the build id


if __name__ == "__main__":
    unittest.main()
