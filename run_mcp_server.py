import os, contextlib, uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp_server_motherduck import build_application

DB   = "/app/data/synthetic_openflights.db"          # data directory is set in Dockerfile
PORT = int(os.environ.get("PORT", 8080))        # Cloud Run injects $PORT

# 1️⃣ Build the core SQL-aware MCP server (read-only for safety).
server, _ = build_application(db_path=DB, read_only=True)

# 2️⃣ Wrap it so HTTP clients can talk to it (ASGI handler).
sess = StreamableHTTPSessionManager(app=server, event_store=None, stateless=True)

async def handler(scope, receive, send):
    await sess.handle_request(scope, receive, send)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with sess.run():
        yield                                        # keep sessions alive

# 3️⃣ Starlette turns that handler into a full ASGI app Uvicorn can serve.
app = Starlette(routes=[Mount("/mcp", app=handler)], lifespan=lifespan)

if __name__ == "__main__":
    print(f"🔥 MCP endpoint → http://0.0.0.0:{PORT}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=PORT)