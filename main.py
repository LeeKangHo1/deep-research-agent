# main.py
"""
Deep Research Chatbot 메인 애플리케이션
FastAPI 기반 웹 서버를 시작합니다.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os

from config.settings import settings


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성 및 설정"""
    
    app = FastAPI(
        title="Deep Research Chatbot",
        description="LangChain과 LangGraph를 활용한 멀티 에이전트 연구 시스템",
        version="1.0.0",
        debug=settings.debug
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발용, 프로덕션에서는 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 로깅 설정
    logger.add(
        settings.log_file,
        rotation="1 day",
        retention="30 days",
        level=settings.log_level
    )
    
    # 데이터 디렉토리 생성
    os.makedirs(os.path.dirname(settings.vector_db_path), exist_ok=True)
    os.makedirs(os.path.dirname(settings.conversation_db_path), exist_ok=True)
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    
    # API 라우터 등록 (추후 구현)
    # from api.chat import router as chat_router
    # app.include_router(chat_router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        """루트 엔드포인트"""
        return {
            "message": "Deep Research Chatbot API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """헬스 체크 엔드포인트"""
        return {"status": "healthy"}
    
    return app


app = create_app()


if __name__ == "__main__":
    logger.info(f"Deep Research Chatbot 서버 시작 - {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )