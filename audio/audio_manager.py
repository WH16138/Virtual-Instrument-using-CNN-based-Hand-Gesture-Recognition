from pathlib import Path
import random
import time


class AudioManager:
    """Optional pygame-based PC audio layer.

    Missing pygame or missing audio files never stops the game. Add files under
    assets/audio and the manager will begin playing them automatically.
    """

    SFX_FILES = {
        "start": ("sfx/start.wav", "sfx/start.ogg"),
        "restart": ("sfx/restart.wav", "sfx/restart.ogg"),
        "card_confirm": ("sfx/card_confirm.wav", "sfx/card_confirm.ogg"),
        "strike": ("sfx/strike.wav", "sfx/strike.ogg"),
        "shot": ("sfx/shot.wav", "sfx/shot.ogg"),
        "guard": ("sfx/guard.wav", "sfx/guard.ogg"),
        "hit": ("sfx/hit.wav", "sfx/hit.ogg"),
        "block": ("sfx/block.wav", "sfx/block.ogg"),
        "heal": ("sfx/heal.wav", "sfx/heal.ogg"),
        "miss": ("sfx/miss.wav", "sfx/miss.ogg"),
        "critical": ("sfx/critical.wav", "sfx/critical.ogg"),
        "wave_start": ("sfx/wave_start.wav", "sfx/wave_start.ogg"),
        "wave_clear": ("sfx/wave_clear.wav", "sfx/wave_clear.ogg"),
        "reward": ("sfx/reward.wav", "sfx/reward.ogg"),
        "reward_open": ("sfx/reward_open.wav", "sfx/reward_open.ogg", "sfx/reward.wav", "sfx/reward.ogg"),
        "reward_apply": ("sfx/reward_apply.wav", "sfx/reward_apply.ogg", "sfx/reward.wav", "sfx/reward.ogg"),
        "augment": ("sfx/augment.wav", "sfx/augment.ogg"),
        "defeat": ("sfx/defeat.wav", "sfx/defeat.ogg"),
        "enemy_attack": ("sfx/enemy_attack.wav", "sfx/enemy_attack.ogg"),
        "enemy_skill": ("sfx/enemy_skill.wav", "sfx/enemy_skill.ogg"),
        "enemy_defend": ("sfx/enemy_defend.wav", "sfx/enemy_defend.ogg"),
    }

    BGM_FILES = {
        "setup": ("bgm/setup_loop.ogg", "bgm/setup_loop.mp3", "bgm/setup_loop.wav"),
        "dungeon": ("bgm/dungeon_loop.ogg", "bgm/dungeon_loop.mp3", "bgm/dungeon_loop.wav"),
    }

    DEFAULT_COOLDOWNS = {
        "hit": 0.08,
        "heal": 0.12,
        "block": 0.12,
        "miss": 0.12,
        "augment": 0.15,
        "card_confirm": 0.18,
    }

    def __init__(self, asset_root="assets/audio", sfx_volume=0.75, bgm_volume=0.35, enabled=True):
        self.asset_root = Path(asset_root)
        self.sfx_volume = float(sfx_volume)
        self.bgm_volume = float(bgm_volume)
        self.requested_enabled = bool(enabled)
        self.enabled = False
        self.pygame = None
        self.sounds = {}
        self.loaded_bgm = {}
        self.current_bgm = None
        self.current_bgm_path = None
        self.last_played_at = {}
        self.reported_missing = set()

        if self.requested_enabled:
            self._initialize()

    def _initialize(self):
        try:
            import pygame
        except Exception as exc:
            print(f"[Audio] pygame unavailable; audio disabled ({exc}).")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
        except Exception as exc:
            print(f"[Audio] mixer init failed; audio disabled ({exc}).")
            return

        self.pygame = pygame
        self.enabled = True
        self._load_sounds()
        if not self.sounds and not self._has_any_bgm_file():
            print("[Audio] No audio assets found under assets/audio; audio layer is ready but silent.")

    def _load_sounds(self):
        if not self.enabled:
            return
        for name, candidates in self.SFX_FILES.items():
            path = self._first_existing(candidates)
            if path is None:
                continue
            try:
                sound = self.pygame.mixer.Sound(str(path))
                sound.set_volume(self.sfx_volume)
                self.sounds[name] = sound
            except Exception as exc:
                print(f"[Audio] Failed to load SFX {path}: {exc}")

    def _first_existing(self, candidates):
        for relative in candidates:
            path = self.asset_root / relative
            if path.exists():
                return path
        return None

    def _has_any_bgm_file(self):
        return any(self._bgm_candidates(name) for name in self.BGM_FILES)

    def play_sfx(self, name, volume=1.0, cooldown=None):
        if not self.enabled:
            return False
        sound = self.sounds.get(name)
        if sound is None:
            return False

        now = time.monotonic()
        min_interval = self.DEFAULT_COOLDOWNS.get(name, 0.04) if cooldown is None else float(cooldown)
        if now - self.last_played_at.get(name, -999.0) < min_interval:
            return False

        self.last_played_at[name] = now
        try:
            sound.set_volume(max(0.0, min(1.0, self.sfx_volume * float(volume))))
            sound.play()
            return True
        except Exception:
            return False

    def start_bgm(self, name="dungeon", fade_ms=600):
        if not self.enabled or self.current_bgm == name:
            return False

        paths = self._bgm_candidates(name)
        if not paths:
            self._report_missing_bgm(name)
            return False
        path = self._choose_bgm_path(name, paths)

        try:
            self.pygame.mixer.music.load(str(path))
            self.pygame.mixer.music.set_volume(max(0.0, min(1.0, self.bgm_volume)))
            self.pygame.mixer.music.play(loops=-1, fade_ms=int(fade_ms))
            self.current_bgm = name
            self.current_bgm_path = path
            return True
        except Exception as exc:
            print(f"[Audio] Failed to start BGM {path}: {exc}")
            self.current_bgm = None
            return False

    def stop_bgm(self, fade_ms=400):
        if not self.enabled or self.current_bgm is None:
            return
        try:
            self.pygame.mixer.music.fadeout(int(fade_ms))
        except Exception:
            try:
                self.pygame.mixer.music.stop()
            except Exception:
                pass
        self.current_bgm = None
        self.current_bgm_path = None

    def _bgm_candidates(self, name):
        cached = self.loaded_bgm.get(name)
        if cached is not None:
            return list(cached)

        paths = []
        seen = set()
        for relative in self.BGM_FILES.get(name, ()):
            path = self.asset_root / relative
            if path.exists() and path not in seen:
                paths.append(path)
                seen.add(path)

        bgm_dir = self.asset_root / "bgm"
        for suffix in (".ogg", ".mp3", ".wav"):
            for path in sorted(bgm_dir.glob(f"{name}_*{suffix}")):
                if path.exists() and path not in seen:
                    paths.append(path)
                    seen.add(path)

        self.loaded_bgm[name] = tuple(paths)
        return list(paths)

    def _choose_bgm_path(self, name, paths):
        if len(paths) <= 1:
            return paths[0]
        candidates = [path for path in paths if path != self.current_bgm_path]
        return random.choice(candidates or paths)

    def update_music(self, game_started, game_state=None):
        if not self.enabled:
            return
        state = (game_state or {}).get("battle_state")
        state_name = getattr(state, "name", str(state or ""))
        if game_started and state_name != "DEFEAT":
            self.start_bgm("dungeon")
        elif state_name == "DEFEAT":
            self.stop_bgm(fade_ms=800)
        else:
            if self._bgm_candidates("setup"):
                self.start_bgm("setup")
            else:
                self.stop_bgm()

    def play_events(self, events, game_state=None):
        if not self.enabled or not events:
            return
        for event in events:
            for sfx_name in self._event_sfx(event):
                self.play_sfx(sfx_name)

    def _event_sfx(self, event):
        event_type = event.get("event_type")
        if event_type == "wave_start":
            return ["wave_start"]
        if event_type == "round_reveal":
            return ["card_confirm"]
        if event_type == "wave_clear":
            return ["wave_clear"]
        if event_type == "reward_select":
            return ["reward_open"]
        if event_type == "reward_apply":
            return ["reward_apply"]
        if event_type == "defeat":
            return ["defeat"]
        if event_type != "action":
            return []

        kind = str(event.get("kind") or "")
        result = str(event.get("result") or "")
        source = str(event.get("source") or "")
        sounds = []

        if result == "block":
            sounds.append("block")
        elif result == "heal":
            sounds.append("heal")
        elif result in ("miss", "heal_failed"):
            sounds.append("miss")
        elif result == "augment" or kind in {"Deep Rest", "Chicken Game", "Cull", "Counter", "Insurance", "Vampire", "Prepared"}:
            sounds.append("augment")

        if source == "player":
            if kind == "Strike":
                sounds.append("strike")
            elif kind == "Shot":
                sounds.append("critical" if event.get("critical") else "shot")
            elif kind == "Guard":
                sounds.append("guard")
        elif source == "enemy":
            if kind == "Attack":
                sounds.append("enemy_attack")
            elif kind == "Skill":
                sounds.append("enemy_skill")
            elif kind == "Defend":
                sounds.append("enemy_defend")

        if int(event.get("damage", 0) or 0) > 0:
            sounds.append("hit")

        deduped = []
        for name in sounds:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _report_missing_bgm(self, name):
        if name in self.reported_missing:
            return
        self.reported_missing.add(name)
        candidates = ", ".join(self.BGM_FILES.get(name, ()))
        print(f"[Audio] BGM '{name}' not found. Expected one of: {candidates} or bgm/{name}_*.ogg/.mp3/.wav")

    def close(self):
        if not self.enabled:
            return
        try:
            self.pygame.mixer.music.stop()
            self.pygame.mixer.quit()
        except Exception:
            pass
        self.enabled = False
