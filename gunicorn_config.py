import multiprocessing
import os

# Get port from environment variable with a fallback to 8000
port = os.environ.get('PORT', '8000')

# Bind to 0.0.0.0 to allow external access
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1  # Use single worker to minimize memory usage
worker_class = "sync"
worker_connections = 50
timeout = 300  # Increased timeout for ML operations
keepalive = 2

# Memory optimization
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

# Process naming
proc_name = "loan_prediction_app"

# Preload app to avoid loading ML models multiple times
preload_app = True

# Debugging help
spew = False
check_config = True

def when_ready(server):
    """Log when server is ready"""
    print(f"Server is ready. Listening on: {bind}")

def on_starting(server):
    """Initialize the application before workers are spawned"""
    print("Initializing application and ML models...")
    print(f"Using port: {port}")
    print(f"Binding to: {bind}")

def post_fork(server, worker):
    """Reduce memory usage after forking"""
    print(f"Worker spawned (pid: {worker.pid})")

def pre_fork(server, worker):
    """Pre-fork handler for initialization"""
    pass

def pre_exec(server):
    """Pre-execution handler"""
    print("Forked child, re-executing.")

# SSL Configuration (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile" 