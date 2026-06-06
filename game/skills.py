class Skill:
    """Combat skill metadata."""

    def __init__(self, name, damage, description, action):
        self.name = name
        self.damage = damage
        self.description = description
        self.action = action


class SkillManager:
    """Map recognized gestures to combat actions."""

    def __init__(self):
        self.skills = {
            "Fist": Skill("Strike", damage=15, description="Close punch strike", action="Strike"),
            "Open_Palm": Skill("Guard", damage=0, description="Block the next incoming hit", action="Guard"),
            "V_Sign": Skill("Arcane Shot", damage=15, description="Focused ranged shot", action="Shot"),
            "Gun_Sign": Skill("Arcane Shot", damage=15, description="Focused ranged shot", action="Shot"),
        }

    def get_skill(self, gesture):
        """Return the combat skill for a gesture, if it has one."""
        return self.skills.get(gesture)

    def get_action_from_gesture(self, gesture):
        """Return the combat action for a gesture.

        OK_Sign is intentionally not mapped here. It is reserved for starting
        the game and should not consume a battle turn.
        """
        skill = self.get_skill(gesture)
        return skill.action if skill is not None else None
