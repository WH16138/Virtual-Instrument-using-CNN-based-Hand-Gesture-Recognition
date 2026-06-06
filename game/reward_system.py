import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardChoice:
    """A selectable run-limited reward definition."""

    reward_id: str
    title: str
    category: str
    description: str = ""
    effects: dict = field(default_factory=dict)

    def to_state(self, index, slot_action):
        return {
            "id": self.reward_id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "slot_index": int(index),
            "slot_action": slot_action,
            "effects": dict(self.effects),
        }


DEFAULT_REWARD_CATALOG = [
    RewardChoice(
        reward_id="vital_rune",
        title="Vital Rune",
        category="stat",
        description="Max HP +15, heal 15",
        effects={"max_hp": 15, "heal": 15},
    ),
    RewardChoice(
        reward_id="battle_trance",
        title="Battle Trance",
        category="stat",
        description="All damage +15%",
        effects={"damage_multiplier": 1.15},
    ),
    RewardChoice(
        reward_id="power_sigil",
        title="Power Sigil",
        category="stat",
        description="Attack power +4",
        effects={"attack_power": 4},
    ),
    RewardChoice(
        reward_id="heavy_strike",
        title="Heavy Strike",
        category="card_upgrade",
        description="Strike damage +4",
        effects={"strike_bonus": 4},
    ),
    RewardChoice(
        reward_id="renewing_guard",
        title="Renewing Guard",
        category="card_upgrade",
        description="Guard recovery +2%p",
        effects={"guard_heal_ratio_bonus": 0.02},
    ),
    RewardChoice(
        reward_id="focused_shot",
        title="Focused Shot",
        category="card_upgrade",
        description="Shot damage +4",
        effects={"shot_bonus": 4},
    ),
    RewardChoice(
        reward_id="double_attack",
        title="Double Attack",
        category="augment",
        description="Strike/Shot 25% extra",
        effects={"augment_flag": "double_attack"},
    ),
    RewardChoice(
        reward_id="cull_the_weak",
        title="Cull the Weak",
        category="augment",
        description="Win matchup: bonus damage",
        effects={"augment_flag": "cull_the_weak"},
    ),
    RewardChoice(
        reward_id="deep_rest",
        title="Deep Rest",
        category="augment",
        description="Low HP Guard heal +50%",
        effects={"augment_flag": "deep_rest"},
    ),
    RewardChoice(
        reward_id="counter_guard",
        title="Counter Guard",
        category="augment",
        description="Guard may counter offense",
        effects={"augment_flag": "counter_guard"},
    ),
    RewardChoice(
        reward_id="chicken_game",
        title="Chicken Game",
        category="augment",
        description="Shot vs Skill always hits",
        effects={"augment_flag": "chicken_game"},
    ),
    RewardChoice(
        reward_id="vampire",
        title="Vampire",
        category="augment",
        description="Heal 15% of dealt damage",
        effects={"augment_flag": "vampire"},
    ),
    RewardChoice(
        reward_id="prepared",
        title="Prepared",
        category="augment",
        description="Wave start: heal missing HP",
        effects={"augment_flag": "prepared"},
    ),
    RewardChoice(
        reward_id="insurance",
        title="Insurance",
        category="augment",
        description="Lose matchup: heal 3% max HP",
        effects={"augment_flag": "insurance"},
    ),
    RewardChoice(
        reward_id="first_strike",
        title="First Strike",
        category="augment",
        description="Wave start: free Strike",
        effects={"augment_flag": "first_strike"},
    ),
]


class RewardManager:
    """Generate and apply run-limited reward choices."""

    CATEGORIES = ("stat", "heal", "card_upgrade", "augment")
    SLOT_ACTIONS = ("Strike", "Guard", "Shot")

    def __init__(self, catalog=None):
        self.catalog = list(DEFAULT_REWARD_CATALOG if catalog is None else catalog)
        self.current_choices = []

    def reset_run(self):
        self.current_choices = []

    def generate_choices(self, count=3, player=None):
        available_catalog = self._available_catalog(player)
        if not available_catalog:
            self.current_choices = []
            return []

        sample_count = min(int(count), len(available_catalog), len(self.SLOT_ACTIONS))
        selected = random.sample(available_catalog, sample_count)
        self.current_choices = [
            reward.to_state(index, self.SLOT_ACTIONS[index])
            for index, reward in enumerate(selected)
        ]
        return [dict(choice) for choice in self.current_choices]

    def _available_catalog(self, player=None):
        owned_augments = set(getattr(player, "augment_flags", set()) or []) if player is not None else set()
        if not owned_augments:
            return list(self.catalog)

        available = []
        for reward in self.catalog:
            augment_flag = reward.effects.get("augment_flag")
            if augment_flag and str(augment_flag) in owned_augments:
                continue
            available.append(reward)
        return available

    def get_current_choices(self):
        return [dict(choice) for choice in self.current_choices]

    def get_choice(self, reward_id):
        for choice in self.current_choices:
            if choice.get("id") == reward_id:
                return dict(choice)
        return None

    def apply_reward(self, player, reward_id):
        choice = self.get_choice(reward_id)
        if choice is None:
            return {"applied": False, "label": "No reward", "choice": None}

        effects = dict(choice.get("effects") or {})
        self._apply_effects(player, effects)
        return {"applied": True, "label": choice.get("title", "Reward"), "choice": choice}

    def _apply_effects(self, player, effects):
        if not effects:
            return

        max_hp_delta = int(effects.get("max_hp", 0) or 0)
        if max_hp_delta:
            player.max_hp = max(1, player.max_hp + max_hp_delta)
            player.hp = min(player.max_hp, player.hp + max(0, max_hp_delta))

        heal = int(effects.get("heal", 0) or 0)
        if heal:
            player.heal(heal)

        heal_percent = float(effects.get("heal_percent", 0.0) or 0.0)
        if heal_percent:
            player.heal(int(round(player.max_hp * heal_percent)))

        player.attack_power += int(effects.get("attack_power", 0) or 0)
        player.strike_bonus += int(effects.get("strike_bonus", 0) or 0)
        player.guard_bonus += int(effects.get("guard_bonus", 0) or 0)
        player.guard_heal_ratio_bonus += float(effects.get("guard_heal_ratio_bonus", 0.0) or 0.0)
        player.shot_bonus += int(effects.get("shot_bonus", 0) or 0)

        damage_multiplier = float(effects.get("damage_multiplier", 1.0) or 1.0)
        heal_multiplier = float(effects.get("heal_multiplier", 1.0) or 1.0)
        player.damage_multiplier *= max(0.0, damage_multiplier)
        player.heal_multiplier *= max(0.0, heal_multiplier)

        augment_flag = effects.get("augment_flag")
        if augment_flag:
            player.augment_flags.add(str(augment_flag))