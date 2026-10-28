"""Deploy the Streamlit dashboard on Modal.

One-time setup (PowerShell):
    uv run modal setup
    uv run modal secret create supabase-dashboard SUPABASE_URL=https://<project>.supabase.co SUPABASE_ANON_KEY=<anon key>
Deploy:
    uv run modal deploy dashboard/modal_app.py
Modal prints the public URL (https://<workspace>--luxury-resale-dashboard-run.modal.run).
"""
import shlex
import subprocess
from pathlib import Path

import modal

APP_FILE = Path(__file__).parent / "app.py"
REMOTE = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("streamlit~=1.50", "supabase>=2.10", "pandas>=2.2", "altair>=5")
    .add_local_file(APP_FILE, REMOTE)
)
app = modal.App("luxury-resale-dashboard", image=image)


@app.function(secrets=[modal.Secret.from_name("supabase-dashboard")], max_containers=1)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=60)
def run():
    cmd = (f"streamlit run {shlex.quote(REMOTE)} --server.port 8000 --server.address 0.0.0.0 "
           "--server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true")
    subprocess.Popen(cmd, shell=True)
