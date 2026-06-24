import os
import streamlit as st

def _secret(*keys, default=None):
    value = st.secrets
    for key in keys:
        try:
            value = value[key]
        except Exception:
            return default
    return value


ENGINE_ADA = _secret("ENGINE_ADA")
GPT_DEFAULT = "3.5"
GPT3_BASE = _secret("services", "gpt", "GPT_BASE", default=st.secrets.get("GPT_BASE"))
GPT3_VERSION = _secret("services", "gpt", "GPT_VERSION", default=st.secrets.get("GPT_VERSION"))
GPT3_KEY = _secret("services", "gpt", "GPT_KEY", default=st.secrets.get("GPT_KEY"))
GPT3_ENGINE = _secret("services", "gpt", "GPT_ENGINE", default=st.secrets.get("GPT_ENGINE"))
GPT4_BASE = _secret("services", "gpt4o", "GPT4o_BASE", default=st.secrets.get('GPT4o_BASE'))
GPT4_VERSION = _secret("services", "gpt4o", "GPT4o_VERSION", default=st.secrets.get('GPT4o_VERSION'))
GPT4_KEY = _secret("services", "gpt4o", "GPT4o_KEY", default=st.secrets.get('GPT4o_KEY'))
GPT4_ENGINE = _secret("services", "gpt4o", "GPT4o_ENGINE", default=st.secrets.get("GPT4o_ENGINE"))

# Gemini secrets
USE_GEMINI = _secret("settings", "USE_GEMINI", default=st.secrets.get("USE_GEMINI", False))
GEMINI_API_KEY = _secret("services", "gemini", "GEMINI_API_KEY", default=st.secrets.get("GEMINI_API_KEY", ""))
GEMINI_CHAT_MODEL = _secret("services", "gemini", "GEMINI_CHAT_MODEL", default=st.secrets.get("GEMINI_CHAT_MODEL", ""))
GEMINI_EMBEDDING_MODEL = _secret("services", "gemini", "GEMINI_EMBEDDING_MODEL", default=st.secrets.get("GEMINI_EMBEDDING_MODEL", ""))

if GPT_DEFAULT == "4":
    GPT_BASE = GPT4_BASE
    GPT_VERSION = GPT4_VERSION
    GPT_KEY = GPT4_KEY
    GPT_ENGINE = GPT4_ENGINE
elif GPT_DEFAULT == "3.5":
    GPT_BASE = GPT3_BASE
    GPT_VERSION = GPT3_VERSION
    GPT_KEY = GPT3_KEY
    GPT_ENGINE = GPT3_ENGINE
else:
    raise ValueError("GPT_DEFAULT must be '3.5' or '4'")
