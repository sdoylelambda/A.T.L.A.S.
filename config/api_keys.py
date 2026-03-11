import keyring
import getpass
import platform
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SERVICE_NAME = config.get("personalize", {}).get("ai_assistant_name", "ATLAS")

# Next steps: have these display in the GUI. Implement reset_api_key, delete_api_key and list_stored_keys.


def get_api_key(provider: str) -> str:
    """
    Retrieve an API key securely.
    If it does not exist, prompt user once and store it securely.
    """
    provider = provider.lower().strip()
    key = keyring.get_password(SERVICE_NAME, provider)

    if key:
        return key

    print(f"\nNo API key found for '{provider}'.")
    key = getpass.getpass(f"Enter your {provider} API key (input hidden): ").strip()

    if not key:
        raise ValueError("No API key provided.")

    keyring.set_password(SERVICE_NAME, provider, key)
    print(f"✓ Stored securely in {platform.system()} keyring.\n")

    return key


def set_api_key(provider: str):
    """
    Manually set or overwrite an API key.
    """
    provider = provider.lower().strip()
    key = getpass.getpass(f"Enter new {provider} API key (input hidden): ").strip()

    if not key:
        raise ValueError("No API key provided.")

    keyring.set_password(SERVICE_NAME, provider, key)
    print(f"✓ {provider} API key updated securely.")


def reset_api_key(provider: str):
    """
    Alias for set_api_key — clearer semantic intent.
    """
    set_api_key(provider)


def delete_api_key(provider: str):
    """
    Remove a stored API key.
    """
    provider = provider.lower().strip()
    try:
        keyring.delete_password(SERVICE_NAME, provider)
        print(f"✓ {provider} API key deleted.")
    except keyring.errors.PasswordDeleteError:
        print(f"No stored key found for '{provider}'.")


def list_stored_keys():
    """
    Informational helper.
    (Keyring does not support listing usernames securely across all OSes.)
    So this only checks common providers.
    """
    common_providers = ["gemini", "claude", "openai"]
    found = []

    for provider in common_providers:
        if keyring.get_password(SERVICE_NAME, provider):
            found.append(provider)

    if not found:
        print("No stored API keys found.")
    else:
        print("Stored API keys:")
        for p in found:
            print(f" - {p}")
