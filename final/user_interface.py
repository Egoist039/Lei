class UI:
    def __init__(self, shelves, layers):
        self.shelves = shelves
        self.layers = list(layers)

    def _ask(self, txt, opts, func=str):
        while True:
            val = input(txt).strip().upper()
            try:
                v = func(val)
                if v in opts: return v
                print(f"  Invalid. Options: {opts}")
            except:
                print("  Format error.")

    def get_task(self):
        print(f" Shelves: {self.shelves}")
        print(f" Layers : {self.layers}")

        print("\n[PICK]")
        pid = self._ask("  Shelf ID: ", self.shelves)
        play = self._ask("  Layer: ", self.layers, int)

        print("\n[PLACE]")
        did = self._ask("  Shelf ID: ", self.shelves)
        dlay = self._ask("  Layer: ", self.layers, int)

        print(f"\nTask: {pid}-{play} -> {did}-{dlay}\n")
        return {'pick': {'id': pid, 'layer': play}, 'place': {'id': did, 'layer': dlay}}