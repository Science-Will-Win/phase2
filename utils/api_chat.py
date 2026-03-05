"""
Unified streaming API chat client for OpenAI, Anthropic, and Google providers.
Framework-independent module for easy migration to FastAPI/Django.
"""
import os
import yaml
from typing import Generator, Dict, List, Optional


_config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(config: dict):
    with open(_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_api_providers() -> dict:
    config = _load_config()
    return config.get("api_providers", {})


def get_active_model() -> dict:
    config = _load_config()
    return config.get("active_model", {"mode": "local", "local_model": "", "api_provider": "", "api_model": ""})


def save_active_model(mode: str, local_model: str = "", api_provider: str = "", api_model: str = ""):
    config = _load_config()
    if "active_model" not in config:
        config["active_model"] = {}
    config["active_model"]["mode"] = mode
    config["active_model"]["local_model"] = local_model
    config["active_model"]["api_provider"] = api_provider
    config["active_model"]["api_model"] = api_model
    _save_config(config)


def get_api_key(provider: str) -> str:
    config = _load_config()
    providers = config.get("api_providers", {})
    return providers.get(provider, {}).get("api_key", "")


def save_api_key(provider: str, api_key: str):
    config = _load_config()
    if "api_providers" not in config:
        config["api_providers"] = {}
    if provider not in config["api_providers"]:
        config["api_providers"][provider] = {}
    config["api_providers"][provider]["api_key"] = api_key
    _save_config(config)


def get_masked_key(api_key: str) -> str:
    if not api_key or len(api_key) < 8:
        return ""
    return api_key[:4] + "..." + api_key[-4:]


def get_api_keys_status() -> dict:
    providers = get_api_providers()
    result = {}
    for name, cfg in providers.items():
        key = cfg.get("api_key", "")
        result[name] = {
            "has_key": bool(key),
            "masked": get_masked_key(key)
        }
    return result


def get_provider_models() -> dict:
    providers = get_api_providers()
    result = {}
    for name, cfg in providers.items():
        result[name] = {
            "models": cfg.get("models", []),
            "has_key": bool(cfg.get("api_key", ""))
        }
    return result


def _convert_messages_openai(history: List[Dict], message: str, system_prompt: str) -> List[Dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def _convert_messages_anthropic(history: List[Dict], message: str) -> List[Dict]:
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def stream_chat(
    provider: str,
    model: str,
    history: List[Dict],
    message: str,
    system_prompt: str = "",
    temperature: float = 0.6,
    max_tokens: int = 32768
) -> Generator[Dict, None, None]:
    """
    Unified streaming chat interface.
    Yields dicts: {"token": str, "done": bool}
    """
    api_key = get_api_key(provider)
    if not api_key:
        yield {"token": "", "done": True, "error": f"API key not configured for {provider}. Set it in Settings > API Keys."}
        return

    try:
        if provider == "openai":
            yield from _stream_openai(api_key, model, history, message, system_prompt, temperature, max_tokens)
        elif provider == "anthropic":
            yield from _stream_anthropic(api_key, model, history, message, system_prompt, temperature, max_tokens)
        elif provider == "google":
            yield from _stream_google(api_key, model, history, message, system_prompt, temperature, max_tokens)
        else:
            yield {"token": "", "done": True, "error": f"Unknown provider: {provider}"}
    except Exception as e:
        yield {"token": "", "done": True, "error": f"API error ({provider}): {str(e)}"}


def _stream_openai(
    api_key: str, model: str, history: List[Dict], message: str,
    system_prompt: str, temperature: float, max_tokens: int
) -> Generator[Dict, None, None]:
    from openai import OpenAI

    providers = get_api_providers()
    base_url = providers.get("openai", {}).get("base_url", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _convert_messages_openai(history, message, system_prompt)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield {"token": chunk.choices[0].delta.content, "done": False}

    yield {"token": "", "done": True}


def _stream_anthropic(
    api_key: str, model: str, history: List[Dict], message: str,
    system_prompt: str, temperature: float, max_tokens: int
) -> Generator[Dict, None, None]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages = _convert_messages_anthropic(history, message)

    with client.messages.stream(
        model=model,
        messages=messages,
        system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
        temperature=temperature,
        max_tokens=max_tokens
    ) as stream:
        for text in stream.text_stream:
            yield {"token": text, "done": False}

    yield {"token": "", "done": True}


def _stream_google(
    api_key: str, model: str, history: List[Dict], message: str,
    system_prompt: str, temperature: float, max_tokens: int
) -> Generator[Dict, None, None]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    gmodel = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt if system_prompt else None,
        generation_config=generation_config
    )

    # Convert history to Gemini format
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = gmodel.start_chat(history=gemini_history)
    response = chat.send_message(message, stream=True)

    for chunk in response:
        if chunk.text:
            yield {"token": chunk.text, "done": False}

    yield {"token": "", "done": True}
