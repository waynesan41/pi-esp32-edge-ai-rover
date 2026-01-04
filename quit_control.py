class QuitControl:
    def __init__(self):
        # Track button state for combo exit.
        self._options_pressed = False
        self._share_pressed = False

    def _exit_if_combo(self):
        if self._options_pressed and self._share_pressed:
            self.stop_motors()
            raise SystemExit("Options + Share pressed; exiting.")

    def on_options_press(self):
        self._options_pressed = True
        self._exit_if_combo()

    def on_options_release(self):
        self._options_pressed = False

    def on_share_press(self):
        self._share_pressed = True
        self._exit_if_combo()

    def on_share_release(self):
        self._share_pressed = False
