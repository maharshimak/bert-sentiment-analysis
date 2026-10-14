"""Optional database readiness helper. Importing does not connect or read secrets."""
import os
import time

def wait_for_db(*, allow_connection=False, attempts=20, interval=2, connect=None):
    if not allow_connection:
        raise RuntimeError("Database connections require explicit allow_connection=True")
    if not 1 <= attempts <= 30 or not 0 <= interval <= 5:
        raise ValueError("Use bounded attempts and interval")
    required = ("DB_USER", "DB_PASSWORD", "DB_NAME")
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("Set DB_USER, DB_PASSWORD and DB_NAME explicitly")
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    for attempt in range(attempts):
        try:
            connection = connect(host=os.environ.get("DB_HOST", "127.0.0.1"), port=int(os.environ.get("DB_PORT", "5432")), user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"], dbname=os.environ["DB_NAME"], connect_timeout=3)
            connection.close()
            return True
        except Exception:
            if attempt+1 < attempts:
                time.sleep(interval)
    raise RuntimeError("Database not ready; connection details were withheld") from None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-connection", action="store_true")
    args = parser.parse_args()
    wait_for_db(allow_connection=args.allow_connection)
