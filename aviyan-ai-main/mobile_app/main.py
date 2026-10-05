from __future__ import annotations

import json
import os
import threading
import urllib.request
import urllib.error
from datetime import datetime

from kivy.app import App
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

Window.clearcolor = (0.035, 0.04, 0.06, 1)

APP_NAME = "AVIYAN"
CREATOR = "ABISHEK BHUSAL"
COMPANY = "AB DEV STUDIO"
ORIGIN = "NEPAL"
DEFAULT_API = os.environ.get("AVIYAN_API_URL", "http://10.0.2.2:8000")


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 15) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class MessageBubble(Label):
    pass


class AVIYANApp(App):
    title = APP_NAME
    status_text = StringProperty("Online")

    def build(self):
        self.api_url = DEFAULT_API.rstrip("/")
        self.user_id = "mobile-user"
        self.messages = []
        root = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(10))

        header = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(10))
        brand = BoxLayout(orientation="vertical")
        brand.add_widget(Label(text="[b]✦ AVIYAN[/b]", markup=True, font_size=dp(24), halign="left"))
        brand.add_widget(Label(text=f"{COMPANY} • {ORIGIN}", font_size=dp(11), color=(0.6,0.65,0.75,1), halign="left"))
        header.add_widget(brand)
        self.status = Label(text="● Checking…", font_size=dp(12), size_hint_x=None, width=dp(100))
        header.add_widget(self.status)
        root.add_widget(header)

        self.scroll = ScrollView(do_scroll_x=False)
        self.chat = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=[dp(4), dp(4)])
        self.chat.bind(minimum_height=self.chat.setter("height"))
        self.scroll.add_widget(self.chat)
        root.add_widget(self.scroll)

        quick = GridLayout(cols=4, size_hint_y=None, height=dp(42), spacing=dp(6))
        for text in ("Reason", "Code", "Image", "Video"):
            b = Button(text=text, background_normal="", background_color=(0.10,0.12,0.18,1), font_size=dp(12))
            b.bind(on_release=lambda btn: self.quick_prompt(btn.text))
            quick.add_widget(b)
        root.add_widget(quick)

        composer = BoxLayout(size_hint_y=None, height=dp(100), spacing=dp(8))
        self.input = TextInput(
            hint_text="Ask AVIYAN anything…",
            multiline=True,
            padding=[dp(12), dp(12)],
            background_normal="",
            background_color=(0.08,0.10,0.15,1),
            foreground_color=(0.95,0.97,1,1),
        )
        composer.add_widget(self.input)
        self.send = Button(text="SEND\n↗", size_hint_x=None, width=dp(84), background_normal="", background_color=(0.18,0.38,0.85,1), bold=True)
        self.send.bind(on_release=self.send_message)
        composer.add_widget(self.send)
        root.add_widget(composer)

        footer = Label(text="Creator: ABISHEK BHUSAL • Founder & Creator, AB DEV STUDIO", font_size=dp(9), color=(0.45,0.48,0.55,1), size_hint_y=None, height=dp(18))
        root.add_widget(footer)

        self.add_message("AVIYAN", "Hello. I’m AVIYAN. Tell me what you want to build, learn, analyze or create.")
        Clock.schedule_once(lambda *_: self.check_health(), 0.2)
        return root

    def add_message(self, role: str, text: str):
        bubble = MessageBubble(
            text=f"[b]{role}[/b]\n{text}",
            markup=True,
            text_size=(Window.width - dp(45), None),
            size_hint_y=None,
            padding=[dp(12), dp(10)],
            halign="left",
            valign="top",
            color=(0.95,0.97,1,1),
        )
        bubble.height = max(dp(52), bubble.texture_size[1] + dp(20))
        self.chat.add_widget(bubble)
        Clock.schedule_once(lambda *_: setattr(self.scroll, "scroll_y", 0), 0)
        Animation(opacity=1, duration=0.18).start(bubble)

    def quick_prompt(self, kind: str):
        prompts = {
            "Reason": "Help me reason through this problem step by step: ",
            "Code": "Help me build and debug this code/project: ",
            "Image": "Create an image prompt for: ",
            "Video": "Create a cinematic video concept and prompt for: ",
        }
        self.input.text = prompts[kind]
        self.input.focus = True

    def set_busy(self, busy: bool):
        self.send.disabled = busy
        self.send.text = "…" if busy else "SEND\n↗"

    def send_message(self, *_):
        text = self.input.text.strip()
        if not text or self.send.disabled:
            return
        self.input.text = ""
        self.add_message("You", text)
        self.messages.append({"role": "user", "content": text})
        self.set_busy(True)
        threading.Thread(target=self._chat_worker, args=(text,), daemon=True).start()

    def _chat_worker(self, text: str):
        try:
            payload = {"model": "aviyan-5b", "messages": self.messages[-20:], "user_id": self.user_id}
            result = post_json(f"{self.api_url}/v1/chat", payload)
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response returned.")
            Clock.schedule_once(lambda *_: self.finish_response(answer), 0)
        except Exception as exc:
            answer = (
                "I’m in offline-safe mode because the AVIYAN backend is not reachable. "
                "Start the backend and set AVIYAN_API_URL to its address.\n\n"
                f"Connection detail: {exc}"
            )
            Clock.schedule_once(lambda *_: self.finish_response(answer), 0)

    def finish_response(self, answer: str):
        self.messages.append({"role": "assistant", "content": answer})
        self.add_message("AVIYAN", answer)
        self.set_busy(False)

    def check_health(self):
        def worker():
            try:
                result = get_json(f"{self.api_url}/health")
                text = "● Online" if result.get("status") == "ok" else "● Ready"
            except Exception:
                text = "● Offline-safe"
            Clock.schedule_once(lambda *_: setattr(self.status, "text", text), 0)
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    AVIYANApp().run()
