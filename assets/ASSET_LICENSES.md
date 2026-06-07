# VisionQuest Asset License Notes

This document tracks non-code assets included in the VisionQuest repository.

Final submission policy:

- Included assets must be either self-made or CC0/public-domain compatible.
- If a CC BY asset is added later, record the title, author, source URL, license version, and modification notes before submission.
- Raw training datasets are not part of the final submission package.

## Summary

| Asset Group | Path | License / Origin | Notes |
|---|---|---|---|
| Action and enemy cards | `assets/cards/*.png` | Self-made or CC0 | UI card images used for player/enemy/reward presentation. |
| 3D models and grounds | `assets/models/*.glb` | Self-made or CC0 | Runtime GLB assets rendered with `trimesh` and `pyrender`. |
| Sound effects | `assets/audio/sfx/*` | Self-made or CC0 | PC-side event SFX. See `assets/audio/README.md` for trigger mapping. |
| Background music | `assets/audio/bgm/*` | Self-made or CC0 | Randomly selected dungeon BGM pool. |
| Generated QR image | `qr_code.png`, `web/qr_code.png` | Generated locally | Runtime/generated helper image, excluded from source submission by `.gitignore`. |

## CC BY Asset Template

Use this table only if a CC BY asset is introduced.

| File | Title | Author | Source URL | License | Changes |
|---|---|---|---|---|---|
| `path/to/asset.ext` |  |  |  | CC BY 4.0 |  |

## External Code Libraries

External code libraries are not copied into this repository. They are installed through `requirements.txt` and credited in the main `README.md`.
