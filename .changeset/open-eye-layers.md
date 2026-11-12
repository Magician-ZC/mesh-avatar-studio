---
"mesh-avatar": minor
---

Show the original eye while it is open. Avatars built with `tools/build-layers.py --open-eyes` carry the optional layers `eye0_open` and `eye1_open`: the illustration's own pixels around each eye, with a hole for the eye ball. While an eye is fully open the WebGL runtime draws its open layer over the cut-out lid and lash layers (it deforms like the base, so head turns, brows and hair sway still apply, and the iris still follows the gaze through the hole), so the resting avatar matches the illustration; it fades out as soon as the lids move or a drawn eye sprite covers the eye. Avatars without these layers render exactly as before, and older runtimes ignore the extra layer names.
