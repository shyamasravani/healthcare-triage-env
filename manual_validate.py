import requests

API_BASE_URL = "https://shyamaDatascientist-healthcare-triage-env.hf.space"

def check_endpoint(path, method="GET", data=None):
    url = f"{API_BASE_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url)
        else:
            r = requests.post(url, json=data or {})
        print(f"{method} {path} -> {r.status_code}")
        print(r.text)
    except Exception as e:
        print(f"Error calling {path}: {e}")

if __name__ == "__main__":
    print("Validating endpoints...")
    check_endpoint("/reset", "POST")
    check_endpoint("/step", "POST", {"decision": "Refer"})
    check_endpoint("/state", "GET")
