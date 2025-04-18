import multiprocessing

# Gunicorn configuration for memory-constrained environments
bind = "0.0.0.0:8000"
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

# SSL Configuration (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"

def post_fork(server, worker):
    """Reduce memory usage after forking"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    """Pre-fork handler for initialization"""
    pass

def pre_exec(server):
    """Pre-execution handler"""
    server.log.info("Forked child, re-executing.") 