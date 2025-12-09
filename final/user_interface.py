class UserInterface:
    def __init__(self, valid_shelves, layers_range):
        """
        :param valid_shelves: 合法的货架ID列表，如 ['A', 'B', 'C', 'D']
        :param layers_range: 层数范围，如 range(1, 4) 代表 1,2,3 层
        """
        self.valid_shelves = valid_shelves
        self.valid_layers = list(layers_range)

    def _get_input(self, prompt, valid_options, cast_func=str):
        while True:
            user_in = input(prompt).strip().upper()
            try:
                val = cast_func(user_in)
                if val in valid_options:
                    return val
                else:
                    print(f"  [Error] Invalid input. Options: {valid_options}")
            except ValueError:
                print(f"  [Error] Invalid format.")

    def get_user_task(self):
        print("\n" + "="*40)
        print(" ROBOTIC ARM TASK SCHEDULER")
        print("="*40)
        print(f" Available Shelves: {self.valid_shelves}")
        print(f" Available Layers : {self.valid_layers}")
        print("-" * 40)

        # 1. 获取拾取信息
        print("\n[PICK TASK]")
        pick_id = self._get_input("  Select Pick Shelf (ID): ", self.valid_shelves)
        pick_layer = self._get_input("  Select Pick Layer (1-3): ", self.valid_layers, int)

        # 2. 获取放置信息
        print("\n[PLACE TASK]")
        place_id = self._get_input("  Select Place Shelf (ID): ", self.valid_shelves)
        place_layer = self._get_input("  Select Place Layer (1-3): ", self.valid_layers, int)

        print("\n" + "="*40)
        print(f" TASK CONFIRMED: Move object from {pick_id}-{pick_layer} to {place_id}-{place_layer}")
        print("="*40 + "\n")

        return {
            'pick': {'id': pick_id, 'layer': pick_layer},
            'place': {'id': place_id, 'layer': place_layer}
        }