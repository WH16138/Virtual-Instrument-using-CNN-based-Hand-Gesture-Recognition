class Skill:
    """Combat skill metadata."""

    def __init__(self, name, damage, description):
        self.name = name
        self.damage = damage
        self.description = description


class SkillManager:
    """Map recognized gestures to combat actions."""

    def __init__(self):
        self.skills = {
            "Fist": Skill("Attack", damage=10, description="Basic punch attack"),
            "Open_Palm": Skill("Defend", damage=0, description="Raise defense for the next hit"),
            "V_Sign": Skill("Power Attack", damage=20, description="Stronger gesture attack"),
        }

    def get_skill(self, gesture):
        """Return the combat skill for a gesture, if it has one."""
        return self.skills.get(gesture)

    def get_action_from_gesture(self, gesture):
        """Return the combat action for a gesture.

        OK_Sign is intentionally not mapped here. It is reserved for starting
        the game and should not consume a battle turn.
        """
        action_map = {
            "Fist": "Attack",
            "Open_Palm": "Defend",
            "V_Sign": "Skill",
        }
        return action_map.get(gesture)
