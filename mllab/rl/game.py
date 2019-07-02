class TwoPlayerFiniteGame(tuple):

    def is_terminal(self):
        """Compute if the state is terminal."""
        return not self.valid_actions()

    @classmethod
    def denormalize_action(cls, action, to):
        _, steps = to.normalized(steps=True)
        for step in steps[::-1]:
            if step == 'flipped':
                action = cls.action_flipped_horizontally(action)
            elif step == 'swapped':
                action = cls.action_swapped(action)
            elif isinstance(step, tuple) and step[0] == 'rotated':
                action = cls.action_rotated(action, -step[1])
            else:
                raise ValueError("unknown step", step)
        return action
