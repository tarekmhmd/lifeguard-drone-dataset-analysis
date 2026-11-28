"""Two-page app, page 1: the AI dashboard, using the process-wide shared hardware service."""
from lifeguard.app.dashboard import page

page(shared=True)
