"""Step 5 - Text-to-speech in a background thread (so the video never freezes)."""
import queue
import threading


class Speaker:
    def __init__(self, enabled=True, rate=170):
        self.enabled = enabled
        self.rate = rate
        self.ok = enabled
        self.busy = False
        self._q = queue.Queue(maxsize=1)
        if enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as e:  # no audio engine available
            print(f"[TTS disabled] {e}")
            self.ok = False
            return
        while True:
            text = self._q.get()
            self.busy = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS error] {e}")
            finally:
                self.busy = False

    def say(self, text):
        """Non-blocking. Returns False if the speaker is busy or disabled."""
        if not self.ok or self.busy:
            return False
        try:
            self._q.put_nowait(text)
            return True
        except queue.Full:
            return False
