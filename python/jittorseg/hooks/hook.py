from .priority import get_priority, Priority


class Hook():
    def __init__(self, runner, priority):
        self.runner = runner
        self._priority = get_priority(priority)

    stages = (
        'before_run', 'after_run',
        'before_epoch', 'after_epoch',
        'before_iter', 'after_iter'
    )

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def priority(self):
        return self._priority

    def before_run(self):
        pass

    def after_run(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass
