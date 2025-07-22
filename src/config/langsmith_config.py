# src/config/langsmith_config.py

"""
LangSmith 모니터링 설정 모듈
LangSmith 연결 및 모니터링 설정을 관리합니다.
"""

import os
from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from .config import Config

def setup_langsmith() -> None:
    """LangSmith 환경 설정을 초기화합니다."""
    # 환경 변수가 이미 설정되어 있으므로 추가 설정은 필요 없음
    # .env 파일에서 자동으로 로드됨
    pass

def get_langsmith_client() -> Client:
    """LangSmith 클라이언트를 반환합니다."""
    return Client(
        api_key=Config.LANGCHAIN_API_KEY,
        api_url=Config.LANGCHAIN_ENDPOINT
    )

def get_callback_manager() -> CallbackManager:
    """LangChain 콜백 매니저를 반환합니다."""
    tracer = LangChainTracer(
        project_name=Config.LANGCHAIN_PROJECT
    )
    return CallbackManager([tracer])

def is_tracing_enabled() -> bool:
    """LangSmith 추적 활성화 여부를 반환합니다."""
    return Config.LANGCHAIN_TRACING_V2 and Config.LANGCHAIN_API_KEY