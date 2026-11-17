---
"mesh-avatar": minor
---

Strands and garments are held only with the new optional rig data: `strands[].reach` (how far a held strand's tip may be pulled from where it hangs, px) and `cloth[].reach` (how far a held hem may be pulled, px, never beyond the garment's `max`). Without it `canGrab` is false and `grab` returns null for that strand or garment, so a client falls back to the drag impulse of avatars built before these fields, which therefore keep their old gestures. While a strand is held its offset is bounded by its reach instead of its sway limit `max`, and after the release that allowance fades while the strand springs back. Avatars without the fields render and sway exactly as before.
