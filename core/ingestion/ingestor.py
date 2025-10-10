from typing import Dict, Iterable, Tuple
from core.clients.client import Client
import gzip, json, os, time, requests

class Ingestor:
    def __init__(self, clients_cfg_path: str, state_dir="var/state", out_dir="var/lake"):
        self.clients = ClientManager(clients_cfg_path)  # thin wrapper around your Client for multiple tenants
        self.state_dir, self.out_dir = state_dir, out_dir
        os.makedirs(state_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    def run_module(self, tenant: str, module: str, fields: Iterable[str], include_rel=True):
        since = self._load_watermark(tenant, module)
        cursor = None
        while True:
            params = {
              "fields": ",".join(fields),
              "since": since or "",
              "page_size": 500,
              "include": "relationships" if include_rel else ""
            }
            if cursor: params["cursor"] = cursor
            resp = self.clients.get(tenant, f"/ai/v1/export/{module}", params=params)  # GET helper
            data = resp["data"] if "data" in resp else resp  # depending on your StandardReturn

            items = data["items"]
            if not items: break

            # write raw snapshot (ndjson.gz)
            path = f"{self.out_dir}/{tenant}/{module}/{int(time.time())}.ndjson.gz"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            # advance cursor/watermark
            cursor = data.get("next_cursor")
            self._save_watermark(tenant, module, data.get("watermark"))
            if not cursor: break

    def _wm_file(self, t, m): return f"{self.state_dir}/{t}__{m}.wm"
    def _load_watermark(self, t,m): 
        p=self._wm_file(t,m); return open(p).read().strip() if os.path.exists(p) else None
    def _save_watermark(self, t,m,wm): 
        if wm: open(self._wm_file(t,m),"w").write(wm)

class ClientManager:
    def __init__(self, cfg_path): self._clients = {}; self._cfg_path = cfg_path
    def _client(self, tenant):
        if tenant not in self._clients:
            self._clients[tenant] = Client(tenant, self._cfg_path)  # your class that loads by name
        return self._clients[tenant]
    def get(self, tenant, path, params):
        c = self._client(tenant)
        # minimal GET that adds HMAC: reuse your sign() but with empty body
        return c.get(path, params=params)  # implement similarly to send_command()
