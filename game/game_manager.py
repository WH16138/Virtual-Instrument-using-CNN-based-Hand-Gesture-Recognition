import time

from game.augment_system import AugmentSystem
from game.battle_system import BattleState, BattleSystem
from game.enemy import Enemy
from game.player import Player
from game.reward_system import RewardManager
from game.skills import SkillManager
from game.wave_manager import WaveManager


class GameManager:
    """High-level game state coordinator."""

    ACTION_HOLD_SECONDS = 2.0
    ROUND_REVEAL_SECONDS = 1.35
    REWARD_HOLD_SECONDS = 2.0
    REWARD_GESTURE_TO_SLOT = {
        "Fist": 0,
        "Open_Palm": 1,
        "V_Sign": 2,
        "Gun_Sign": 2,
    }

    def __init__(self, turn_delay_seconds=1.2, input_repeat_seconds=0.9):
        self.player = Player(max_hp=100)
        self.enemy = Enemy(max_hp=80)
        self.skill_manager = SkillManager()
        self.wave_manager = WaveManager()
        self.reward_manager = RewardManager()
        self.augment_system = AugmentSystem()
        self.battle_system = BattleSystem(self.player, self.enemy, self.skill_manager, self.augment_system)

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
        self.round_reveal = self._empty_round_reveal()
        self.reward_selection = self._empty_reward_selection()

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

    def _empty_round_reveal(self):
        return {
            "active": False,
            "player_action": None,
            "enemy_action": None,
            "started_at": 0.0,
            "progress": 0.0,
            "required_seconds": self.ROUND_REVEAL_SECONDS,
        }

    def _empty_reward_selection(self):
        return {
            "active": False,
            "choices": [],
            "selected_index": None,
            "selected_gesture": None,
            "action": None,
            "started_at": 0.0,
            "hold_progress": 0.0,
            "progress": 0.0,
            "required_seconds": self.REWARD_HOLD_SECONDS,
        }

    def _reset_action_hold(self):
        self.action_hold = self._empty_action_hold()

    def _clear_round_reveal(self):
        self.round_reveal = self._empty_round_reveal()

    def _reset_reward_selection(self):
        self.reward_selection = self._empty_reward_selection()

    def _clear_reward_hold(self):
        self.reward_selection["selected_index"] = None
        self.reward_selection["selected_gesture"] = None
        self.reward_selection["action"] = None
        self.reward_selection["started_at"] = 0.0
        self.reward_selection["hold_progress"] = 0.0
        self.reward_selection["progress"] = 0.0

    def _begin_round_reveal(self, now, player_action, enemy_action):
        if not self.battle_system.begin_round_reveal():
            return False
        self.round_reveal = {
            "active": True,
            "player_action": player_action,
            "enemy_action": enemy_action,
            "started_at": now,
            "progress": 0.0,
            "required_seconds": self.ROUND_REVEAL_SECONDS,
        }
        self.next_action_time = now + self.ROUND_REVEAL_SECONDS
        self.phase_label = f"Reveal: {player_action} vs {enemy_action}"
        self._push_event(
            "round_reveal",
            f"{player_action} vs {enemy_action}",
            source="center",
            kind="Reveal",
            target="center",
            player_action=player_action,
            enemy_action=enemy_action,
        )
        return True

    def _begin_reward_selection(self, now):
        choices = self.reward_manager.generate_choices(3)
        if not choices:
            self._reset_reward_selection()
            self._push_event(
                "reward_skip",
                "No reward choices configured",
                target="center",
                wave=self.wave_manager.current_wave,
            )
            self._start_next_wave(now)
            return False

        self.battle_system.state = BattleState.REWARD_SELECT
        self.reward_selection = self._empty_reward_selection()
        self.reward_selection["active"] = True
        self.reward_selection["choices"] = choices
        self.phase_label = "Choose a reward"
        self.next_action_time = now + 0.25
        self._push_event(
            "reward_select",
            "Choose a reward",
            target="center",
            wave=self.wave_manager.current_wave,
            choices=choices,
        )
        return True

    def start_game(self):
        """Start an infinite wave run."""
        self.wave_manager.reset_run()
        self.reward_manager.reset_run()
        self.battle_system.start_battle()
        now = time.monotonic()
        self._start_next_wave(now)
        self.last_input_gesture = None
        self.last_input_time = 0.0
        self._reset_action_hold()
        self._clear_round_reveal()
        self._reset_reward_selection()

    def _push_event(self, event_type, label, **payload):
        event = {
            "event_type": event_type,
            "label": label,
            "time": time.monotonic(),
            **payload,
        }
        self.events.append(event)
        self.event_history.append(event)
        self.event_history = self.event_history[-14:]

    def _push_battle_events(self, events):
        for event in events:
            payload = dict(event)
            label = payload.pop("label", payload.get("kind", "Action"))
            self._push_event("action", label, **payload)

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
        self._clear_round_reveal()
        self._reset_reward_selection()
        self._push_event(
            "wave_start",
            self.phase_label,
            target="center",
            wave=self.wave_manager.current_wave,
            enemy_type=enemy_type.name,
            difficulty=difficulty,
        )
        augment_events = self.augment_system.on_wave_start({"player": self.player, "enemy": self.enemy})
        if augment_events:
            self.battle_system.last_damage += sum(int(event.get("damage", 0) or 0) for event in augment_events)
            self._push_battle_events(augment_events)
            if not self.enemy.is_alive:
                self.battle_system.state = BattleState.WAVE_CLEAR
                self.phase_label = "Wave clear"
                self.next_action_time = now + 1.7
                self._push_event(
                    "wave_clear",
                    f"Wave {self.wave_manager.current_wave} clear",
                    target="center",
                    wave=self.wave_manager.current_wave,
                )

    def process_gesture(self, gesture_info):
        """Process recognized gestures during action or reward selection phases."""
        if self.battle_system.state == BattleState.REWARD_SELECT:
            self._reset_action_hold()
            return self._process_reward_gesture(gesture_info)

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

        if self.action_hold["gesture"] != gesture:
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

        enemy_action = self.enemy.choose_action()
        self.last_input_gesture = gesture
        self.last_input_time = now
        self._reset_action_hold()
        return self._begin_round_reveal(now, action, enemy_action)

    def _process_reward_gesture(self, gesture_info):
        now = time.monotonic()
        if now < self.next_action_time:
            self._clear_reward_hold()
            return False

        choices = self.reward_selection.get("choices") or []
        if not self.reward_selection.get("active") or not choices:
            self._start_next_wave(now)
            return False

        gesture = gesture_info.get("smoothed_gesture", "Unknown")
        confidence = gesture_info.get("confidence", 0.0)
        slot_index = self.REWARD_GESTURE_TO_SLOT.get(gesture)
        if confidence <= 0.6 or slot_index is None or slot_index >= len(choices):
            self._clear_reward_hold()
            return False

        choice = choices[slot_index]
        slot_action = choice.get("slot_action")
        if self.reward_selection.get("selected_gesture") != gesture:
            self.reward_selection["selected_index"] = slot_index
            self.reward_selection["selected_gesture"] = gesture
            self.reward_selection["action"] = slot_action
            self.reward_selection["started_at"] = now
            self.reward_selection["hold_progress"] = 0.0
            self.reward_selection["progress"] = 0.0
            return False

        elapsed = now - float(self.reward_selection.get("started_at", now) or now)
        progress = min(1.0, elapsed / self.REWARD_HOLD_SECONDS)
        self.reward_selection["hold_progress"] = progress
        self.reward_selection["progress"] = progress
        self.reward_selection["action"] = slot_action
        if progress < 1.0:
            return False

        result = self.reward_manager.apply_reward(self.player, choice.get("id"))
        self._push_event(
            "reward_apply",
            result.get("label") or choice.get("title", "Reward"),
            target="player",
            reward=result.get("choice") or choice,
            applied=result.get("applied", False),
        )
        self.last_input_gesture = gesture
        self.last_input_time = now
        self._reset_reward_selection()
        self._start_next_wave(now)
        return True

    def update(self):
        """Advance timed wave, reveal, reward, and result phases."""
        state = self.battle_system.state
        now = time.monotonic()

        if state == BattleState.WAVE_INTRO:
            self._reset_action_hold()
            self._clear_round_reveal()
            self._reset_reward_selection()
            if now >= self.next_action_time:
                self.battle_system.begin_player_turn()
                self.enemy.prepare_action_weights()
                self.phase_label = "Choose an action"
                self.next_action_time = now + 0.25
                self._push_event("ready", "Action ready", target="player")
            return

        if state == BattleState.WAVE_CLEAR:
            self._reset_action_hold()
            self._clear_round_reveal()
            if now >= self.next_action_time:
                self._begin_reward_selection(now)
            return

        if state == BattleState.REWARD_SELECT:
            self._reset_action_hold()
            self._clear_round_reveal()
            if not self.reward_selection.get("active") or not self.reward_selection.get("choices"):
                self._start_next_wave(now)
            return

        if state != BattleState.ROUND_REVEAL:
            return

        if not self.round_reveal.get("active"):
            self.battle_system.state = BattleState.PLAYER_TURN
            self.enemy.prepare_action_weights()
            self.phase_label = "Choose an action"
            return

        self.round_reveal["progress"] = min(
            1.0,
            (now - self.round_reveal["started_at"]) / max(self.ROUND_REVEAL_SECONDS, 0.001),
        )
        if now < self.next_action_time:
            return

        player_action = self.round_reveal.get("player_action")
        enemy_action = self.round_reveal.get("enemy_action")
        self._clear_round_reveal()
        events = self.battle_system.resolve_simultaneous_round(player_action, enemy_action)
        self._push_battle_events(events)

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
        elif self.battle_system.state == BattleState.WAVE_CLEAR:
            self.phase_label = "Wave clear"
            self.next_action_time = now + 1.7
            self._push_event(
                "wave_clear",
                f"Wave {self.wave_manager.current_wave} clear",
                target="center",
                wave=self.wave_manager.current_wave,
            )
        else:
            self.enemy.prepare_action_weights()
            self.phase_label = "Choose an action"
            self.next_action_time = now + self.turn_delay_seconds

        self.last_input_gesture = None
        self._reset_action_hold()

    def reset_game(self):
        """Reset all game state."""
        self.battle_system.reset_battle()
        self.wave_manager.finish_run()
        self.wave_manager.reset_run()
        self.reward_manager.reset_run()
        self.next_action_time = 0.0
        self.last_input_gesture = None
        self.last_input_time = 0.0
        self.events.clear()
        self.event_history.clear()
        self.phase_label = "Prepare the board"
        self._reset_action_hold()
        self._clear_round_reveal()
        self._reset_reward_selection()

    def _turn_delay_remaining(self):
        remaining = self.next_action_time - time.monotonic()
        return max(0.0, remaining)

    def _enemy_action_hint(self):
        return self.enemy.get_action_probabilities()

    def _reward_selection_state(self):
        state = dict(self.reward_selection)
        state["choices"] = [dict(choice) for choice in self.reward_selection.get("choices", [])]
        return state

    def get_game_state(self):
        """Return the current state used by HUD and game logic."""
        state = self.battle_system.state
        delay_remaining = self._turn_delay_remaining()
        round_reveal = dict(self.round_reveal)
        return {
            "player": {
                "hp": self.player.hp,
                "max_hp": self.player.max_hp,
                "is_defending": self.player.is_defending,
                "growth": {
                    "attack_power": self.player.attack_power,
                    "strike_bonus": self.player.strike_bonus,
                    "guard_bonus": self.player.guard_bonus,
                    "guard_heal_ratio_bonus": self.player.guard_heal_ratio_bonus,
                    "shot_bonus": self.player.shot_bonus,
                    "damage_multiplier": self.player.damage_multiplier,
                    "heal_multiplier": self.player.heal_multiplier,
                    "augment_flags": sorted(self.player.augment_flags),
                    "augments": self.augment_system.get_active_states(self.player),
                },
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
            "can_select_reward": state == BattleState.REWARD_SELECT and delay_remaining <= 0.0,
            "action_selection": dict(self.action_hold),
            "reward_selection": self._reward_selection_state(),
            "enemy_action_hint": self._enemy_action_hint(),
            "round_reveal": round_reveal,
            "enemy_preview": {
                "active": False,
                "action": None,
                "progress": 0.0,
                "required_seconds": self.ROUND_REVEAL_SECONDS,
            },
        }