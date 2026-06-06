from enum import Enum
import random


class BattleState(Enum):
    """Battle state."""

    WAITING = 0
    WAVE_INTRO = 1
    PLAYER_TURN = 2
    ROUND_REVEAL = 3
    ENEMY_TURN = 4
    WAVE_CLEAR = 5
    DEFEAT = 6
    VICTORY = 7


class BattleSystem:
    """Simultaneous card battle system."""

    PLAYER_HEAL_RATIO = 0.08
    PLAYER_HEAL_MIN = 6
    ENEMY_HEAL_RATIO = 0.08
    ENEMY_HEAL_MIN = 4
    SHOT_CRITICAL_MULTIPLIER = 2
    ENEMY_SKILL_MULTIPLIER = 2

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
        self.reset_defenses()
        self.state = BattleState.WAVE_INTRO
        self.last_action = f"Wave {wave_number}: {self.enemy.name}"
        self.last_damage = 0

    def begin_player_turn(self):
        self.state = BattleState.PLAYER_TURN
        self.turn_count += 1

    def begin_round_reveal(self):
        if self.state == BattleState.PLAYER_TURN:
            self.state = BattleState.ROUND_REVEAL
            return True
        return False

    def resolve_simultaneous_round(self, player_action, enemy_action):
        """Resolve one simultaneous card reveal and return feedback events."""
        if self.state not in (BattleState.ROUND_REVEAL, BattleState.PLAYER_TURN):
            return []
        if player_action not in ("Strike", "Guard", "Shot"):
            return []
        if enemy_action not in ("Attack", "Defend", "Skill"):
            return []

        self.reset_defenses()
        self.last_action = f"{player_action} vs {enemy_action}"
        self.last_damage = 0
        events = []

        def add_event(source, kind, label, target, damage=0, heal=0, result=None, critical=False):
            event = {
                "source": source,
                "kind": kind,
                "label": label,
                "target": target,
                "damage": int(damage or 0),
                "heal": int(heal or 0),
                "result": result,
                "critical": bool(critical),
            }
            events.append(event)
            return event

        def damage_enemy(amount, kind, label=None, critical=False):
            actual = self.enemy.take_damage(int(amount))
            self.last_damage += actual
            add_event(
                "player",
                kind,
                label or kind,
                "enemy",
                damage=actual,
                result="critical" if critical else "damage",
                critical=critical,
            )

        def damage_player(amount, kind, label=None, critical=False):
            actual = self.player.take_damage(int(amount))
            self.last_damage += actual
            add_event(
                "enemy",
                kind,
                label or kind,
                "player",
                damage=actual,
                result="critical" if critical else "damage",
                critical=critical,
            )

        def heal_player(amount, kind="Guard"):
            before = self.player.hp
            self.player.heal(int(amount))
            actual = self.player.hp - before
            add_event("player", kind, f"+{actual}" if actual > 0 else "FULL", "player", heal=actual, result="heal")

        def heal_enemy(amount, kind="Defend"):
            before = self.enemy.hp
            self.enemy.heal(int(amount))
            actual = self.enemy.hp - before
            add_event("enemy", kind, f"+{actual}" if actual > 0 else "FULL", "enemy", heal=actual, result="heal")

        def miss(source, kind, target, label="MISS", result="miss"):
            add_event(source, kind, label, target, result=result)

        def block(source, kind, target):
            add_event(source, kind, "BLOCK", target, result="block")

        strike_damage = 12
        shot_damage = 22 * self.SHOT_CRITICAL_MULTIPLIER
        player_heal = max(self.PLAYER_HEAL_MIN, int(round(self.player.max_hp * self.PLAYER_HEAL_RATIO)))
        enemy_heal = max(self.ENEMY_HEAL_MIN, int(round(self.enemy.max_hp * self.ENEMY_HEAL_RATIO)))
        enemy_skill_damage = int(self.enemy.base_damage * self.ENEMY_SKILL_MULTIPLIER)

        if player_action == "Strike":
            if enemy_action == "Defend":
                block("enemy", "Defend", "enemy")
                heal_enemy(enemy_heal)
            else:
                damage_enemy(strike_damage, "Strike")
        elif player_action == "Guard":
            if enemy_action == "Skill":
                miss("player", "Guard", "player", label="HEAL FAIL", result="heal_failed")
            else:
                heal_player(player_heal)
        elif player_action == "Shot":
            if enemy_action == "Defend":
                chance = 1.0
            elif enemy_action == "Attack":
                chance = 0.25
            else:
                chance = 0.50
            if random.random() <= chance:
                damage_enemy(shot_damage, "Shot", label="CRIT", critical=True)
            else:
                miss("player", "Shot", "enemy")

        if enemy_action == "Attack":
            if player_action == "Guard":
                block("player", "Guard", "player")
            else:
                damage_player(self.enemy.base_damage, "Attack")
        elif enemy_action == "Defend":
            if player_action == "Shot":
                miss("enemy", "Defend", "enemy", label="HEAL FAIL", result="heal_failed")
            elif player_action == "Strike":
                pass
            else:
                heal_enemy(enemy_heal)
        elif enemy_action == "Skill":
            if player_action == "Guard":
                chance = 1.0
            elif player_action == "Strike":
                chance = 0.25
            else:
                chance = 0.50
            if random.random() <= chance:
                damage_player(enemy_skill_damage, "Skill", label="CRIT", critical=True)
            else:
                miss("enemy", "Skill", "player")

        if not self.player.is_alive:
            self.state = BattleState.DEFEAT
        elif not self.enemy.is_alive:
            self.state = BattleState.WAVE_CLEAR
        else:
            self.state = BattleState.PLAYER_TURN
            self.turn_count += 1

        return events

    def handle_player_action(self, gesture):
        """Legacy single-action entrypoint kept for compatibility."""
        action = self.skill_manager.get_action_from_gesture(gesture)
        if action is None:
            return None
        enemy_action = self.enemy.choose_action()
        events = self.resolve_simultaneous_round(action, enemy_action)
        return events[0] if events else None

    def enemy_turn(self, action=None):
        """Legacy enemy turn entrypoint kept for compatibility."""
        if self.state != BattleState.ENEMY_TURN:
            return None
        action = action or self.enemy.choose_action()
        event = {
            "source": "enemy",
            "kind": action,
            "label": action,
            "damage": 0,
            "target": "player",
        }
        if action == "Attack":
            event["damage"] = self.player.take_damage(self.enemy.base_damage)
        elif action == "Skill":
            event["damage"] = self.player.take_damage(int(self.enemy.base_damage * self.ENEMY_SKILL_MULTIPLIER))
        elif action == "Defend":
            self.enemy.heal(max(self.ENEMY_HEAL_MIN, int(round(self.enemy.max_hp * self.ENEMY_HEAL_RATIO))))
            event["result"] = "heal"
        self.state = BattleState.DEFEAT if not self.player.is_alive else BattleState.PLAYER_TURN
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
            BattleState.ROUND_REVEAL,
            BattleState.ENEMY_TURN,
            BattleState.WAVE_CLEAR,
        )
