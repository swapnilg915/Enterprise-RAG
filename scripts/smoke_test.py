import os
import json
import urllib.request


def main() -> None:
    url = os.getenv("SERVICE_URL", "http://127.0.0.1:8004")
    with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
        print(response.read().decode())
    request = urllib.request.Request(
        f"{url}/api/v1/ask",
        data=json.dumps(
            {"question": "Which company focuses on governed enterprise AI?"}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    if not payload["citations"]:
        raise RuntimeError("Smoke test returned no citations")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
