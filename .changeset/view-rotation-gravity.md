---
"mesh-avatar": minor
---

Add the optional rig fields `view.rotation` and `physics.gravity`, and `toClientPoint(x, y)` on `MeshAvatar` (WebGL runtime only). `view.rotation` (radians, default 0) draws the whole avatar rotated about the image centre, fitting the rotated bounding box, so a figure that was turned upright to be bound is shown the way its picture was; the rig, the deformers and `toImagePoint` stay in unrotated source pixels, and lighting stays off on a rotated view. `physics.gravity` (radians, default π/2, straight down the image) is the direction hair strands, buns and garments hang and sway in; the breeze and the head-roll correction run across it. `toClientPoint` is the inverse of `toImagePoint`. Rigs without the fields keep their original code paths and render exactly as before.
