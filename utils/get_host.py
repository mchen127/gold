import socket

def get_hostname():
    """Get the hostname of the current machine."""
    return socket.gethostname()
