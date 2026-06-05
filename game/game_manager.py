import time

from game.battle_system import BattleState, BattleSystem
from game.enemy import Enemy
from game.player import Player
from game.skills import SkillManager
from game.wave_manager import WaveManager


class GameManager:
    """High-level game state coordinator."""

    ACTION_HOLD_SECONDS = 2.0
    ENEMY_PREVIEW_SECONDS = 2.0

    def __init__(self, turn_delay_seconds=1.2, input_repeat_seconds=0.9):
        self.player = Player(max_hp=100)
        self.enemy = Enemy(max_hp=80)
        self.skill_manager = SkillManager()
        self.wave_manager = WaveManager()
        self.battle_system = BattleSystem(self.player, self.enemy, self.skill_manager)

        self.player_pos = (200, 150)
        self.enemy_pos = (200, 50)
        self.turn_delay_seconds = turn_delay_seconds
        self.input_repeat_seconds = input_repeat_seconds
        self.next_action_time = 0.0
        self.last_input_gesture = None
        self.last_input_time = 0.0
        self.events = []
        self.event_history = []
        self.phase_label = "Prepare the board"
        self.action_hold = self._empty_action_hold()
        self.enemy_preview_action = None
        self.enemy_preview_started_at = 0.0
        self.enemy_preview_progress = 0.0

    def _empty_action_hold(self):
        return {
            "gesture": None,
            "action": None,
            "label": None,
            "started_at": 0.0,
            "progress": 0.0,
            "required_seconds": self.ACTION_HOLD_SECONDS,
            "active": False,
        }

    def _reset_action_hold(self):
        self.action_hold = self._empty_action_hold()

    def _begin_enemy_preview(self, now):
        self.enemy_preview_action = self.enemy.choose_action()
        self.enemy_preview_started_at = now
        self.enemy_preview_progress = 0.0
        self.next_action_time = now + self.ENEMY_PREVIEW_SECONDS
        self.phase_label = "Enemy choosing"
        self._push_event(
            "enemy_preview",
            self.enemy_preview_action,
            source="enemy",
            kind=self.enemy_preview_action,
            target="enemy",
        )

    def _clear_enemy_preview(self):
        self.enemy_preview_action = None
        self.enemy_preview_started_at = 0.0
        self.enemy_preview_progress = 0.0

    def start_game(self):
        """Start an infinite wave run."""
        self.wave_manager.reset_run()
        self.battle_system.start_battle()
        now = time.monotonic()
        self._start_next_wave(now)
        self.last_input_gesture = None
        self.last_input_time = 0.0
        self._reset_action_hold()
        self._clear_enemy_preview()

    def _push_event(self, event_type, label, **payload):
        event = {
            "event_type": event_type,
            "label": label,
            "time": time.monotonic(),
            **payload,
        }
        self.events.append(event)
        self.event_history.append(event)
        self.event_history = self.event_history[-10:]

    def consume_events(self):
        events = list(self.events)
        self.events.clear()
        return events

    def _start_next_wave(self, now=None):
        now = time.monotonic() if now is None else now
        enemy_type, difficulty = self.wave_manager.next_wave()
        self.battle_system.start_wave(enemy_type, difficulty, self.wave_manager.current_wave)
        self.phase_label = f"Wave {self.wave_manager.current_wave}: {enemy_type.name}"
        self.next_action_time = now + 1.45
        self._push_event(
            "wave_start",
            self.phase_label,
            target="center",
            wave=self.wave_manager.current_wave,
            enemy_type=enemy_type.name,
        )

    def process_gesture(self, gesture_info):
        """Process a recognized gesture during the player's turn."""
        if not self.battle_system.is_battle_active:
            self._reset_action_hold()
            return False
        if self.battle_system.state != BattleState.PLAYER_TURN:
            self._reset_action_hold()
            return False

        now = time.monotonic()
        if now < self.next_action_time:
            self._reset_action_hold()
            return False

        gesture = gesture_info.get("smoothed_gesture", "Unknown")
        confidence = gesture_info.get("confidence", 0.0)
        if confidence <= 0.6 or gesture == "Unknown":
            self._reset_action_hold()
            return False

        skill = self.skill_manager.get_skill(gesture)
        action = self.skill_manager.get_action_from_gesture(gesture)
        if skill is None or action is None:
            self._reset_action_hold()
            return False

        if self.action_hold["action"] != action:
            self.action_hold = {
                "gesture": gesture,
                "action": action,
                "label": skill.name,
                "started_at": now,
                "progress": 0.0,
                "required_seconds": self.ACTION_HOLD_SECONDS,
                "active": True,
            }
            return False

        self.action_hold["gesture"] = gesture
        elapsed = now - self.action_hold["started_at"]
        self.action_hold["progress"] = min(1.0, elapsed / self.ACTION_HOLD_SECONDS)
        if self.action_hold["progress"] < 1.0:
            return False

        event = self.battle_system.handle_player_action(gesture)
        self._reset_action_hold()
        if event is not None:
            self.last_input_gesture = gesture
            self.last_input_time = now
            payload = dict(event)
            label = payload.pop("label")
            self._push_event("action", label, **payload)
            if self.battle_system.state == BattleState.ENEMY_TURN:
                self._begin_enemy_preview(now)
            else:
                self.phase_label = "Wave clear"
                self.next_action_time = now + self.turn_delay_seconds
            if self.battle_system.state == BattleState.WAVE_CLEAR:
                self._push_event(
                    "wave_clear",
                    f"Wave {self.wave_manager.current_wave} clear",
                    target="center",
                    wave=self.wave_manager.current_wave,
                )
                self.next_action_time = now + 1.7
        return event is not None

    def update(self):
        """Advance timed wave and enemy phases."""
        state = self.battle_system.state
        now = time.monotonic()

        if state == BattleState.WAVE_INTRO:
            self._reset_action_hold()
            self._clear_enemy_preview()
            if now >= self.next_action_time:
                self.battle_system.begin_player_turn()
                self.phase_label = "Your turn"
                self.next_action_time = now + 0.25
                self._push_event("ready", "Action ready", target="player")
            return

        if state == BattleState.WAVE_CLEAR:
            self._reset_action_hold()
            self._clear_enemy_preview()
            if now >= self.next_action_time:
                self._start_next_wave(now)
            return

        if state != BattleState.ENEMY_TURN:
            return

        if self.enemy_preview_action is None:
            self._begin_enemy_preview(now)
            return

        self.enemy_preview_progress = min(
            1.0,
            (now - self.enemy_preview_started_at) / max(self.ENEMY_PREVIEW_SECONDS, 0.001),
        )
        if now < self.next_action_time:
            return

        preview_action = self.enemy_preview_action
        self._clear_enemy_preview()
        event = self.battle_system.enemy_turn(preview_action)
        if event is not None:
            self.player.set_defense(False)
            payload = dict(event)
            label = payload.pop("label")
            self._push_event("action", label, **payload)
            if self.battle_system.state == BattleState.DEFEAT:
                self.wave_manager.finish_run()
                self.phase_label = "Defeat"
                self._push_event(
                    "defeat",
                    f"Defeated at wave {self.wave_manager.current_wave}",
                    target="center",
                    wave=self.wave_manager.current_wave,
                    best_wave=self.wave_manager.best_wave,
                )
            else:
                self.phase_label = "Your turn"
                self.next_action_time = time.monotonic() + self.turn_delay_seconds
            self.last_input_gesture = None
            self._reset_action_hold()

    def reset_game(self):
        """Reset all game state."""
        self.battle_system.reset_battle()
        self.wave_manager.finish_run()
        self.wave_manager.reset_run()
        self.next_action_time = 0.0
        self.last_input_gesture = None
        self.last_input_time = 0.0
        self.events.clear()
        self.event_history.clear()
        self.phase_label = "Prepare the board"
        self._reset_action_hold()
        self._clear_enemy_preview()

    def _turn_delay_remaining(self):
        remaining = self.next_action_time - time.monotonic()
        return max(0.0, remaining)

    def get_game_state(self):
        """Return the current state used by HUD and game logic."""
        state = self.battle_system.state
        delay_remaining = self._turn_delay_remaining()
        return {
            "player": {
                "hp": self.player.hp,
                "max_hp": self.player.max_hp,
                "is_defending": self.player.is_defending,
            },
            "enemy": {
                "hp": self.enemy.hp,
                "max_hp": self.enemy.max_hp,
                "is_defending": self.enemy.is_defending,
                "name": self.enemy.name,
                "color": self.enemy.color,
                "model_path": self.enemy.model_path,
                "ground_model_path": self.enemy.ground_model_path,
            },
            "battle_state": state,
            "phase_label": self.phase_label,
            "last_action": self.battle_system.last_action,
            "last_damage": self.battle_system.last_damage,
            "turn_count": self.battle_system.turn_count,
            "wave": self.wave_manager.current_wave,
            "best_wave": self.wave_manager.best_wave,
            "enemy_type": self.wave_manager.current_enemy_type.name,
            "difficulty": self.wave_manager.global_difficulty_multiplier,
            "events": list(self.event_history),
            "turn_delay_remaining": delay_remaining,
            "can_act": state == BattleState.PLAYER_TURN and delay_remaining <= 0.0,
            "action_selection": dict(self.action_hold),
            "enemy_preview": {
                "active": state == BattleState.ENEMY_TURN and self.enemy_preview_action is not None,
                "action": self.enemy_preview_action,
                "progress": self.enemy_preview_progress,
                "required_seconds": self.ENEMY_PREVIEW_SECONDS,
            },
        }
