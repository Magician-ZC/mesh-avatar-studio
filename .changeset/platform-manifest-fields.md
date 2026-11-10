---
"mesh-avatar": minor
---

Validate and preserve the optional `finalCapabilities` and `platform` manifest blocks and the optional `hitZones` rig field. Avatars without them are unchanged; the blocks also survive repacking with `--no-source`.
