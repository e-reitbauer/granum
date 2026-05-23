from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional


SOCKET_NAME = "granum.sock"


def socket_path(granum_dir: Path) -> Path:
    return granum_dir / SOCKET_NAME


# ------------------------------------------------------------------
# Server side
# ------------------------------------------------------------------

async def start_ipc_server(handler, granum_dir: Path) -> None:
    sock = socket_path(granum_dir)
    sock.unlink(missing_ok=True)

    server = await asyncio.start_unix_server(_make_handler(handler), path=str(sock))
    sock.chmod(0o600)

    async with server:
        await server.serve_forever()


def _make_handler(handler):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.readline()
            request = json.loads(data.decode())
            method = request.get("method")
            params = request.get("params", {})

            try:
                result = await handler(method, params)
                response = {"result": result}
            except Exception as e:
                response = {"error": str(e)}

            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    return handle


# ------------------------------------------------------------------
# Client side
# ------------------------------------------------------------------

def ipc_call(granum_dir: Path, method: str, params: dict) -> Optional[Any]:
    """Synchronous IPC call. Returns result or None if server unavailable."""
    sock = socket_path(granum_dir)
    if not sock.exists():
        return None

    import socket as _socket

    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect(str(sock))
            payload = json.dumps({"method": method, "params": params}) + "\n"
            s.sendall(payload.encode())

            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk

            response = json.loads(buf.split(b"\n")[0].decode())
            if "error" in response:
                return None
            return response["result"]
    except Exception:
        return None
