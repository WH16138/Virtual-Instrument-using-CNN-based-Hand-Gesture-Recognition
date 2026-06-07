# VisionQuest Audio Assets

Put PC playback audio files here. The game runs even when this folder is empty.

The audio layer uses `pygame.mixer` through `audio/AudioManager`. SFX can be `.wav` or `.ogg`; BGM can be `.ogg`, `.mp3`, or `.wav`. Short WAV/OGG files are recommended for SFX, and loopable OGG files are recommended for BGM. If multiple BGM files match the same category, one is selected randomly when that BGM starts. The recommended location is `assets/audio/bgm/`, but the loader also searches other `assets/audio/` subfolders for matching `setup_*` or `dungeon_*` files to tolerate misplaced assets.

## BGM Files

| Path | When It Plays | Notes |
|---|---|---|
| `assets/audio/bgm/setup_loop.ogg` | Optional setup/waiting ambience before gameplay starts. | If missing, setup remains silent. Also accepts `.mp3` or `.wav`. |
| `assets/audio/bgm/setup_*.ogg` | Optional random setup BGM pool. | Also accepts `.mp3` or `.wav`. One file is chosen randomly when setup BGM starts. |
| `assets/audio/bgm/dungeon_loop.ogg` | Main loop during active gameplay, including wave intro, player card selection, reveal, reward selection, and wave progression. | Stops/fades out on defeat. Also accepts `.mp3` or `.wav`. |
| `assets/audio/bgm/dungeon_*.ogg` | Optional random dungeon BGM pool. | Also accepts `.mp3` or `.wav`. One file is chosen randomly when dungeon BGM starts. |

## SFX Files

| Path | Trigger Situation | Suggested Sound |
|---|---|---|
| `assets/audio/sfx/start.wav` | Board registration completes and the run starts after holding `OK_Sign`. | Magical gate open, start chime. |
| `assets/audio/sfx/restart.wav` | Defeat screen restart completes after holding `OK_Sign`. | Short reset/respawn chime. |
| `assets/audio/sfx/card_focus.wav` | A player action card or reward card first becomes the active hold target, including when the recognized card changes. Falls back to `card_confirm` if missing. | Soft hover, card focus, cursor tick. |
| `assets/audio/sfx/card_confirm.wav` | Player card is confirmed and the simultaneous reveal begins. | Card flip, selection lock, UI confirm. |
| `assets/audio/sfx/strike.wav` | Player `Strike` action event. | Punch, slash, blunt hit windup. |
| `assets/audio/sfx/shot.wav` | Player `Shot` action event when it is not marked critical. | Magic bolt, projectile launch. |
| `assets/audio/sfx/critical.wav` | Player `Shot` action event when marked critical. | Strong magic impact or crit sparkle. |
| `assets/audio/sfx/guard.wav` | Player `Guard` action event. | Shield raise, barrier form. |
| `assets/audio/sfx/hit.wav` | Any event with positive damage, after player or enemy action sounds. | Impact, damage hit. |
| `assets/audio/sfx/block.wav` | Any event whose result is `block`. | Shield block, metallic/parry impact. |
| `assets/audio/sfx/heal.wav` | Any event whose result is `heal`. | Healing shimmer. |
| `assets/audio/sfx/miss.wav` | Any event whose result is `miss` or `heal_failed`. | Whiff, fizzle, failed magic. |
| `assets/audio/sfx/wave_start.wav` | New wave starts and enemy is configured. | Enemy spawn, wave bell, portal pulse. |
| `assets/audio/sfx/wave_clear.wav` | Enemy dies and wave clear event is emitted. | Victory stinger, clear fanfare. |
| `assets/audio/sfx/reward_open.wav` | Reward selection opens after wave clear and the three reward cards appear. | Treasure chest open, reward fanfare, card spread. |
| `assets/audio/sfx/reward_apply.wav` | A reward card is selected and applied after the hold completes. | Item pickup, upgrade confirm, rune absorb. |
| `assets/audio/sfx/reward.wav` | Legacy fallback used when `reward_open` or `reward_apply` is missing. | Generic reward chime. |
| `assets/audio/sfx/augment.wav` | Augment effect event such as Deep Rest, Chicken Game, Cull, Counter, Insurance, Vampire, or Prepared. `agument.wav` is accepted as a typo fallback. | Passive proc, rune activation. |
| `assets/audio/sfx/defeat.wav` | Player dies and defeat event is emitted. | Defeat sting. |
| `assets/audio/sfx/enemy_attack.wav` | Enemy `Attack` action event. | Monster swing, claw, bite. |
| `assets/audio/sfx/enemy_skill.wav` | Enemy `Skill` action event. | Dark spell, monster special. |
| `assets/audio/sfx/enemy_defend.wav` | Enemy `Defend` action event. | Enemy guard, armor, shell. |

Each SFX path also accepts an `.ogg` alternative with the same base name. For example, `assets/audio/sfx/strike.ogg` works if `strike.wav` is not present.

## Cooldown Behavior

Some frequently repeated SFX are rate-limited to avoid noisy stacking:

| SFX | Default Cooldown |
|---|---:|
| `hit` | 0.08 s |
| `heal` | 0.12 s |
| `block` | 0.12 s |
| `miss` | 0.12 s |
| `augment` | 0.15 s |
| `card_confirm` | 0.18 s |

## License Tracking

Track downloaded or generated assets in a project-level asset license document, not only here. Recommended file:

```text
assets/ASSET_LICENSES.md
```

For CC BY assets, record title, author, source URL, license version, and changes. CC0 or self-made assets can be grouped more simply, but keeping source notes is still recommended.
