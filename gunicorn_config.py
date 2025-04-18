import multiprocessing
import os

# Get port from environment variable or use default
port = os.environ.get('PORT', '10000')
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

# Process naming
proc_name = "loan_prediction_app"

# Preload app to avoid loading ML models multiple times
preload_app = True

# SSL Configuration (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"

def on_starting(server):
    """Initialize the application before workers are spawned"""
    server.log.info("Initializing application and ML models...")

def post_fork(server, worker):
    """Reduce memory usage after forking"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    """Pre-fork handler for initialization"""
    pass

def pre_exec(server):
    """Pre-execution handler"""
    server.log.info("Forked child, re-executing.") 