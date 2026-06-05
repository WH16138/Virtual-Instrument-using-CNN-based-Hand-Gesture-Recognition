from enum import Enum


class BattleState(Enum):
    """Battle state."""

    WAITING = 0
    PLAYER_TURN = 1
    ENEMY_TURN = 2
    VICTORY = 3
    DEFEAT = 4


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
        """Start a fresh battle."""
        self.player.reset()
        self.enemy.reset()
        self.state = BattleState.PLAYER_TURN
        self.last_action = "Battle started"
        self.last_damage = 0
        self.turn_count = 1

    def handle_player_action(self, gesture):
        """Handle one player action from a recognized combat gesture."""
        if self.state != BattleState.PLAYER_TURN:
            return False

        action = self.skill_manager.get_action_from_gesture(gesture)
        skill = self.skill_manager.get_skill(gesture)

        if action is None or skill is None:
            return False

        self.last_action = f"Player: {action}"
        self.last_damage = 0

        if action == "Attack":
            self.last_damage = skill.damage
            self.enemy.take_damage(self.last_damage)
        elif action == "Defend":
            self.player.set_defense(True)
        elif action == "Skill":
            self.last_damage = skill.damage
            self.enemy.take_damage(self.last_damage)

        if self.enemy.is_alive:
            self.state = BattleState.ENEMY_TURN
        else:
            self.state = BattleState.VICTORY

        return True

    def enemy_turn(self):
        """Run one enemy action."""
        if self.state != BattleState.ENEMY_TURN:
            return False

        action = self.enemy.choose_action()
        self.last_action = f"Enemy: {action}"
        self.last_damage = 0

        if action == "Attack":
            self.last_damage = self.player.take_damage(8)
        elif action == "Defend":
            self.enemy.set_defense(True)
        elif action == "Skill":
            self.last_damage = self.player.take_damage(15)

        if not self.player.is_alive:
            self.state = BattleState.DEFEAT
        else:
            self.state = BattleState.PLAYER_TURN

        self.turn_count += 1
        return True

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
        return self.state in (BattleState.PLAYER_TURN, BattleState.ENEMY_TURN)
