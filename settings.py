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


# Azure OpenAI settings
GPT_BASE    = _secret("services", "gpt", "AZURE_OPENAI_ENDPOINT")
GPT_VERSION = _secret("services", "gpt", "AZURE_OPENAI_API_VERSION")
GPT_KEY     = _secret("services", "gpt", "AZURE_OPENAI_API_KEY")
GPT_ENGINE  = _secret("services", "gpt", "AZURE_OPENAI_DEPLOYMENT")

# Aliases used by embeddings.py (same single Azure deployment)
GPT3_BASE    = GPT_BASE
GPT3_VERSION = GPT_VERSION
GPT3_KEY     = GPT_KEY
ENGINE_ADA   = GPT_ENGINE

# Gemini settings
USE_GEMINI             = _secret("settings", "USE_GEMINI", default=False)
GEMINI_API_KEY         = _secret("services", "gemini", "GEMINI_API_KEY", default="")
GEMINI_CHAT_MODEL      = _secret("services", "gemini", "GEMINI_CHAT_MODEL", default="")
GEMINI_EMBEDDING_MODEL = _secret("services", "gemini", "GEMINI_EMBEDDING_MODEL", default="")
