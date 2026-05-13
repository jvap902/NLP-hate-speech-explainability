import time
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.markdown import Markdown

# Message buffer
messages = []

class LivePanel():
    def __init__(self, title, color="blue"):
        self.title = title
        self.messages = []
        self.color = color
        
        self.live = Live(self, refresh_per_second=4, auto_refresh=True)
        
    def start(self):
        """Inicia a exibição ao vivo no terminal."""
        self.live.start()

    def stop(self):
        """Para a exibição e limpa o terminal."""
        self.live.stop()
        
    def addMessage(self, message):
        self.messages.append(message)
        
    def __rich__(self):        
        message_group = Group(*[Markdown(m) for m in self.messages])
        return Panel(message_group, title=self.title, border_style=self.color)