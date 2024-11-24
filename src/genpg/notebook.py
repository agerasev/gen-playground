from IPython.display import DisplayHandle, display


class Display:
    def __init__(self):
        self.displays: list[DisplayHandle] = []
        self.index = 0

    def clear(self):
        self.index = 0

    def update(self, obj):
        while self.index >= len(self.displays):
            self.displays.append(display(display_id=True))
        self.displays[self.index].update(obj, update=True)
        self.index += 1
