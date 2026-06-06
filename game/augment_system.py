import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentDefinition:
    flag: str
    label: str
    short_label: str
    description: str

    def to_state(self):
        return {
            "flag": self.flag,
            "label": self.label,
            "short_label": self.short_label,
            "description": self.description,
        }


AUGMENT_DEFINITIONS = [
    AugmentDefinition("double_attack", "이중공격", "Double", "Strike/Shot can trigger again."),
    AugmentDefinition("cull_the_weak", "약자멸시", "Cull", "Bonus damage on matchup wins."),
    AugmentDefinition("deep_rest", "깊은휴식", "Rest", "Guard heals more under 30% HP."),
    AugmentDefinition("counter_guard", "반격", "Counter", "Guard can counter enemy offense."),
    AugmentDefinition("chicken_game", "치킨게임", "Chicken", "Shot always hits in Skill vs Skill."),
    AugmentDefinition("vampire", "뱀파이어", "Vampire", "Heal from damage dealt."),
    AugmentDefinition("prepared", "만반의 준비", "Ready", "Heal missing HP on wave start."),
    AugmentDefinition("insurance", "보험금", "Insure", "Heal when losing matchup."),
    AugmentDefinition("first_strike", "선제 공격", "First", "Strike immediately on wave start."),
]

AUGMENT_BY_FLAG = {definition.flag: definition for definition in AUGMENT_DEFINITIONS}

PLAYER_ACTION_TO_CARD = {
    "Strike": "Attack",
    "Guard": "Defend",
    "Shot": "Skill",
}

CARD_ADVANTAGE = {
    ("Attack", "Skill"),
    ("Defend", "Attack"),
    ("Skill", "Defend"),
}


class AugmentSystem:
    """Hook-style run augment resolver."""

    DOUBLE_ATTACK_CHANCE = 0.25
    CULL_BONUS_RATIO = 0.40
    DEEP_REST_HP_THRESHOLD = 0.30
    DEEP_REST_HEAL_MULTIPLIER = 1.50
    COUNTER_CHANCE = 0.30
    VAMPIRE_HEAL_RATIO = 0.15
    PREPARED_MISSING_HEAL_RATIO = 0.30
    INSURANCE_MAX_HP_HEAL_RATIO = 0.03

    def get_active_states(self, player):
        flags = sorted(getattr(player, "augment_flags", set()))
        states = []
        for flag in flags:
            definition = AUGMENT_BY_FLAG.get(flag)
            if definition is None:
                states.append({"flag": flag, "label": flag, "short_label": flag[:8], "description": ""})
            else:
                states.append(definition.to_state())
        return states

    @staticmethod
    def has(player, flag):
        return flag in getattr(player, "augment_flags", set())

    @staticmethod
    def player_card(player_action):
        return PLAYER_ACTION_TO_CARD.get(player_action)

    def player_wins_matchup(self, player_action, enemy_action):
        return (self.player_card(player_action), enemy_action) in CARD_ADVANTAGE

    def player_loses_matchup(self, player_action, enemy_action):
        return (enemy_action, self.player_card(player_action)) in CARD_ADVANTAGE

    @staticmethod
    def compute_strike_damage(player):
        attack_power = max(0, int(getattr(player, "attack_power", 15)))
        strike_bonus = int(getattr(player, "strike_bonus", 0))
        damage_multiplier = float(getattr(player, "damage_multiplier", 1.0))
        return max(0, int(round((attack_power + strike_bonus) * damage_multiplier)))

    def modify_player_heal(self, context, heal_amount):
        player = context["player"]
        if (
            self.has(player, "deep_rest")
            and context.get("player_action") == "Guard"
            and player.hp_percentage < self.DEEP_REST_HP_THRESHOLD
        ):
            boosted = max(int(heal_amount), int(round(float(heal_amount) * self.DEEP_REST_HEAL_MULTIPLIER)))
            if boosted > heal_amount:
                context["add_event"]("player", "Deep Rest", "REST", "player", result="augment")
            return boosted
        return heal_amount

    def modify_player_shot_chance(self, context, chance):
        player = context["player"]
        if (
            self.has(player, "chicken_game")
            and context.get("player_action") == "Shot"
            and context.get("enemy_action") == "Skill"
        ):
            context["add_event"]("player", "Chicken Game", "ALL IN", "enemy", result="augment")
            return 1.0
        return chance

    def after_round_resolve(self, context):
        player = context["player"]
        if not getattr(player, "augment_flags", None):
            return []

        events = context["events"]
        player_action = context.get("player_action")
        enemy_action = context.get("enemy_action")
        attack_power = max(0, int(context.get("attack_power", getattr(player, "attack_power", 15))))
        enemy = context["enemy"]
        damage_enemy = context["damage_enemy"]
        heal_player = context["heal_player"]
        add_event = context["add_event"]
        pre_augment_event_count = int(context.get("pre_augment_event_count", len(events)))

        primary_player_damage_events = [
            event
            for event in events[:pre_augment_event_count]
            if event.get("source") == "player"
            and event.get("target") == "enemy"
            and event.get("kind") in ("Strike", "Shot")
            and int(event.get("damage", 0) or 0) > 0
        ]

        if self.has(player, "double_attack") and primary_player_damage_events and enemy.is_alive:
            if random.random() <= self.DOUBLE_ATTACK_CHANCE:
                primary = primary_player_damage_events[0]
                damage_enemy(
                    int(primary.get("damage", 0) or 0),
                    primary.get("kind", "Strike"),
                    label="DOUBLE",
                    critical=bool(primary.get("critical")),
                )

        if self.has(player, "cull_the_weak") and self.player_wins_matchup(player_action, enemy_action) and enemy.is_alive:
            bonus = max(1, int(round(attack_power * self.CULL_BONUS_RATIO)))
            damage_enemy(bonus, "Cull", label="CULL")

        if (
            self.has(player, "counter_guard")
            and player_action == "Guard"
            and enemy_action in ("Attack", "Skill")
            and enemy.is_alive
        ):
            if random.random() <= self.COUNTER_CHANCE:
                if enemy_action == "Skill":
                    counter_damage = int(context.get("enemy_skill_damage", enemy.base_damage * 2))
                else:
                    counter_damage = int(enemy.base_damage)
                damage_enemy(counter_damage, "Counter", label="COUNTER")

        if self.has(player, "insurance") and self.player_loses_matchup(player_action, enemy_action):
            heal = max(1, int(round(player.max_hp * self.INSURANCE_MAX_HP_HEAL_RATIO)))
            heal_player(heal, "Insurance")

        if self.has(player, "vampire"):
            damage_total = sum(
                int(event.get("damage", 0) or 0)
                for event in events
                if event.get("source") == "player"
                and event.get("target") == "enemy"
                and int(event.get("damage", 0) or 0) > 0
            )
            if damage_total > 0:
                heal = max(1, int(round(damage_total * self.VAMPIRE_HEAL_RATIO)))
                heal_player(heal, "Vampire")

        return events

    def on_wave_start(self, context):
        player = context["player"]
        enemy = context["enemy"]
        if not getattr(player, "augment_flags", None):
            return []

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

        if self.has(player, "prepared"):
            missing_hp = max(0, player.max_hp - player.hp)
            heal = int(round(missing_hp * self.PREPARED_MISSING_HEAL_RATIO))
            if heal > 0:
                before = player.hp
                player.heal(heal)
                actual = player.hp - before
                add_event("player", "Prepared", f"+{actual}" if actual > 0 else "FULL", "player", heal=actual, result="heal")

        if self.has(player, "first_strike") and enemy.is_alive:
            damage = self.compute_strike_damage(player)
            actual = enemy.take_damage(damage)
            add_event("player", "Strike", "FIRST", "enemy", damage=actual, result="damage")

        return events