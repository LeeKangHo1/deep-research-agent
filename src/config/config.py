# src/config/config.py

"""
설정 관리 모듈
환경 변수 및 애플리케이션 설정을 관리합니다.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# .env 파일 로드
load_dotenv()

class Config:
    """애플리케이션 설정 클래스"""
    
    # API 키
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # LangSmith 설정
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "deep-research-chatbot")
    
    # 애플리케이션 설정
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 검색 설정
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # 대화 설정
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "10"))
    
    @classmethod
    def validate(cls) -> bool:
        """필수 설정 값 검증"""
        required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
        
        for key in required_keys:
            if not getattr(cls, key):
                print(f"경고: {key} 환경 변수가 설정되지 않았습니다.")
                return False
        
        return True
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """모든 설정을 딕셔너리로 반환"""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith("__") and not callable(value)
        }

# 설정 검증
if not Config.validate():
    print("경고: 일부 필수 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")