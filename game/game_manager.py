import time

from game.battle_system import BattleState, BattleSystem
from game.enemy import Enemy
from game.player import Player
from game.skills import SkillManager


class GameManager:
    """High-level game state coordinator."""

    def __init__(self, turn_delay_seconds=1.2, input_repeat_seconds=0.9):
        self.player = Player(max_hp=100)
        self.enemy = Enemy(max_hp=80)
        self.skill_manager = SkillManager()
        self.battle_system = BattleSystem(self.player, self.enemy, self.skill_manager)

        self.player_pos = (200, 150)
        self.enemy_pos = (200, 50)
        self.turn_delay_seconds = turn_delay_seconds
        self.input_repeat_seconds = input_repeat_seconds
        self.next_action_time = 0.0
        self.last_input_gesture = None
        self.last_input_time = 0.0

    def start_game(self):
        """Start the battle and allow the first player action after a short beat."""
        self.battle_system.start_battle()
        now = time.monotonic()
        self.next_action_time = now + 0.4
        self.last_input_gesture = None
        self.last_input_time = 0.0

    def process_gesture(self, gesture_info):
        """Process a recognized gesture during the player's turn."""
        if not self.battle_system.is_battle_active:
            return False
        if self.battle_system.state != BattleState.PLAYER_TURN:
            return False

        now = time.monotonic()
        if now < self.next_action_time:
            return False

        gesture = gesture_info.get("smoothed_gesture", "Unknown")
        confidence = gesture_info.get("confidence", 0.0)
        if confidence <= 0.6 or gesture == "Unknown":
            return False

        if (
            gesture == self.last_input_gesture
            and now - self.last_input_time < self.input_repeat_seconds
        ):
            return False

        handled = self.battle_system.handle_player_action(gesture)
        if handled:
            self.last_input_gesture = gesture
            self.last_input_time = now
            self.next_action_time = now + self.turn_delay_seconds
        return handled

    def update(self):
        """Advance delayed enemy turns."""
        if self.battle_system.state != BattleState.ENEMY_TURN:
            return

        now = time.monotonic()
        if now < self.next_action_time:
            return

        acted = self.battle_system.enemy_turn()
        if acted:
            self.battle_system.reset_defenses()
            self.next_action_time = time.monotonic() + self.turn_delay_seconds
            self.last_input_gesture = None

    def reset_game(self):
        """Reset all game state."""
        self.battle_system.reset_battle()
        self.next_action_time = 0.0
        self.last_input_gesture = None
        self.last_input_time = 0.0

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
            },
            "battle_state": state,
            "last_action": self.battle_system.last_action,
            "last_damage": self.battle_system.last_damage,
            "turn_count": self.battle_system.turn_count,
            "turn_delay_remaining": delay_remaining,
            "can_act": state == BattleState.PLAYER_TURN and delay_remaining <= 0.0,
        }
