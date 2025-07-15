import os

def log_environment_variables(log_message):
    """Log environment variables with sensitive data obscured"""
    log_message("="*60)
    log_message("Environment Variables:")
    for key in sorted(os.environ.keys()):
        value = os.environ[key]
        if "SECRET" in key or "PASSWORD" in key or "KEY" in key:
            if len(value) > 10:
                value = value[:10] + "..." + value[-4:]
            else:
                value = "***"
        elif "BUCKET" in key or "REGION" in key or "ENDPOINT" in key:
            # Show these in full
            pass
        elif len(value) > 50:
            value = value[:50] + "..."
        log_message(f"  {key}={value}")
    log_message("="*60)