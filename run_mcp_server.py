import os
import sys
import contextlib
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp_server_motherduck import build_application


DB_PATH = "/app/data/synthetic_openflights.db"
PORT = int(os.environ.get("PORT", 8080))

server, _ = build_application(db_path=DB_PATH, read_only=True)
sess = StreamableHTTPSessionManager(app=server, event_store=None, stateless=True)

async def handler(scope, receive, send):
    await sess.handle_request(scope, receive, send)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with sess.run():
        yield

app = Starlette(routes=[Mount("/mcp", app=handler)], lifespan=lifespan)


if __name__ == "__main__":
    # --- Robust Pre-flight Checks ---
    # These print to stderr and force a flush to guarantee they appear in logs.
    sys.stderr.write("\n--- Starting Pre-flight Checks ---\n")
    sys.stderr.flush()

    # 1. Check current working directory
    cwd = os.getcwd()
    sys.stderr.write(f"Current Working Directory: {cwd}\n")
    sys.stderr.flush()

    # 2. Check for the database file
    sys.stderr.write(f"Checking for database at absolute path: {DB_PATH}\n")
    if not os.path.exists(DB_PATH):
        sys.stderr.write(f"❌ FATAL ERROR: Database file NOT FOUND at '{DB_PATH}'\n")
        try:
            # If the DB isn't there, let's see what IS in the directory.
            sys.stderr.write(f"Contents of /app: {os.listdir('/app')}\n")
        except Exception as e:
            sys.stderr.write(f"Could not list contents of /app: {e}\n")
        sys.stderr.flush()
        exit(1) # CRASH THE CONTAINER.
    else:
        file_size = os.path.getsize(DB_PATH)
        sys.stderr.write(f"✅ SUCCESS: Database file found at '{DB_PATH}' with size {file_size} bytes.\n")
        sys.stderr.flush()
    
    sys.stderr.write("--- Pre-flight Checks Complete. Starting Uvicorn server. ---\n")
    sys.stderr.flush()

    uvicorn.run(app, host="0.0.0.0", port=PORT)