---
"mesh-avatar": minor
---

Add `toImagePoint(clientX, clientY)` and `impulse(target, vx, vy)` to `MeshAvatar` (WebGL runtime only). `toImagePoint` maps viewport coordinates to source pixels through the canvas position, border, CSS scaling, device pixel ratio, padding and fit; the renderer's internal canvas-to-source mapping now uses separate horizontal and vertical device pixel ratios. `impulse` gives the body, the head, a hair strand or a garment an instantaneous velocity in source px/s, bounded for each kind of target, which the existing springs and pendulums let settle. Avatars render exactly as before unless `impulse` is called.
