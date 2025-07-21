# config/settings.py
"""
애플리케이션 설정 관리
환경 변수와 기본 설정값들을 관리합니다.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings
from datetime import timedelta


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # Flask 설정
    flask_env: str = "development"
    flask_debug: bool = True
    secret_key: str = "dev-secret-key-change-in-production"
    
    # JWT 설정
    jwt_secret_key: str = "jwt-secret-key-change-in-production"
    jwt_access_token_expires: int = 3600  # 초 단위
    
    # MySQL 데이터베이스 설정
    database_url: str = "mysql+pymysql://username:password@localhost:3306/deep_research_chatbot"
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "username"
    mysql_password: str = "password"
    mysql_database: str = "deep_research_chatbot"
    
    # AI 모델 API 키
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # 검색 도구 API 키
    tavily_api_key: Optional[str] = None
    
    # LangSmith 모니터링 설정
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "deep-research-chatbot"
    
    # Chroma 벡터 데이터베이스 설정
    chroma_persist_directory: str = "./chroma_db"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    
    # CORS 설정
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    
    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 5000
    
    # 에이전트 설정
    max_research_depth: int = 3
    max_sources_per_query: int = 10
    agent_timeout: int = 300  # 초
    
    # 검색 설정
    search_results_limit: int = 20
    similarity_threshold: float = 0.7
    
    # 로깅 설정
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def cors_origins_list(self) -> List[str]:
        """CORS origins를 리스트로 반환"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def jwt_access_token_expires_delta(self) -> timedelta:
        """JWT 토큰 만료 시간을 timedelta로 반환"""
        return timedelta(seconds=self.jwt_access_token_expires)


# 전역 설정 인스턴스
settings = Settings()