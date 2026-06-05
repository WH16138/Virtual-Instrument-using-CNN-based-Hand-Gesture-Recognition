from enum import Enum


class BattleState(Enum):
    """Battle state."""

    WAITING = 0
    WAVE_INTRO = 1
    PLAYER_TURN = 2
    ENEMY_TURN = 3
    WAVE_CLEAR = 4
    DEFEAT = 5
    VICTORY = 6


class BattleSystem:
    """Turn-based battle system."""

    def __init__(self, player, enemy, skill_manager):
        self.player = player
        self.enemy = enemy
        self.skill_manager = skill_manager
        self.state = BattleState.WAITING
        self.last_action = None
        self.last_damage = 0
        self.turn_count = 0

    def start_battle(self):
        """Start a fresh run."""
        self.player.reset()
        self.state = BattleState.WAITING
        self.last_action = "Run started"
        self.last_damage = 0
        self.turn_count = 0

    def start_wave(self, enemy_type, difficulty_multiplier, wave_number):
        """Configure the next enemy and enter the wave intro phase."""
        self.enemy.configure(enemy_type, difficulty_multiplier)
        self.enemy.set_defense(False)
        self.player.set_defense(False)
        self.state = BattleState.WAVE_INTRO
        self.last_action = f"Wave {wave_number}: {self.enemy.name}"
        self.last_damage = 0

    def begin_player_turn(self):
        self.state = BattleState.PLAYER_TURN
        self.turn_count += 1

    def handle_player_action(self, gesture):
        """Handle one player action from a recognized combat gesture."""
        if self.state != BattleState.PLAYER_TURN:
            return None

        action = self.skill_manager.get_action_from_gesture(gesture)
        skill = self.skill_manager.get_skill(gesture)

        if action is None or skill is None:
            return None

        self.last_action = f"Hero: {skill.name}"
        self.last_damage = 0
        event = {
            "source": "player",
            "kind": action,
            "label": skill.name,
            "damage": 0,
            "target": "enemy",
        }

        if action in ("Strike", "Shot"):
            self.last_damage = self.enemy.take_damage(skill.damage)
            event["damage"] = self.last_damage
        elif action == "Guard":
            self.player.set_defense(True)
            event["target"] = "player"

        if self.enemy.is_alive:
            self.state = BattleState.ENEMY_TURN
        else:
            self.state = BattleState.WAVE_CLEAR

        return event

    def enemy_turn(self, action=None):
        """Run one enemy action."""
        if self.state != BattleState.ENEMY_TURN:
            return None

        action = action or self.enemy.choose_action()
        self.last_action = f"{self.enemy.name}: {action}"
        self.last_damage = 0
        event = {
            "source": "enemy",
            "kind": action,
            "label": action,
            "damage": 0,
            "target": "player",
        }

        if action == "Attack":
            self.last_damage = self.player.take_damage(self.enemy.base_damage)
            event["damage"] = self.last_damage
        elif action == "Defend":
            self.enemy.set_defense(True)
        elif action == "Skill":
            self.last_damage = self.player.take_damage(int(self.enemy.base_damage * 1.65))
            event["damage"] = self.last_damage

        if not self.player.is_alive:
            self.state = BattleState.DEFEAT
        else:
            self.state = BattleState.PLAYER_TURN

        return event

    def reset_defenses(self):
        """Clear temporary defense states."""
        self.player.set_defense(False)
        self.enemy.set_defense(False)

    def reset_battle(self):
        """Reset the battle to the waiting state."""
        self.state = BattleState.WAITING
        self.player.reset()
        self.enemy.reset()
        self.last_action = None
        self.last_damage = 0
        self.turn_count = 0

    @property
    def is_battle_active(self):
        return self.state in (
            BattleState.WAVE_INTRO,
            BattleState.PLAYER_TURN,
            BattleState.ENEMY_TURN,
            BattleState.WAVE_CLEAR,
        )
