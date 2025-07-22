# src/app.py

"""
애플리케이션 진입점
Deep Research Chatbot 애플리케이션의 메인 진입점입니다.
"""

import os
import sys
from dotenv import load_dotenv
from config import Config, setup_langsmith
from utils.logging_utils import setup_logger

# 로거 설정
logger = setup_logger("app")

def setup_environment():
    """환경 설정을 초기화합니다."""
    # .env 파일 로드
    load_dotenv()
    
    # 설정 검증
    if not Config.validate():
        logger.warning("일부 필수 환경 변수가 설정되지 않았습니다.")
    
    # LangSmith 설정
    setup_langsmith()
    
    logger.info("환경 설정이 완료되었습니다.")

def main():
    """애플리케이션 메인 함수"""
    logger.info("Deep Research Chatbot 시작")
    
    # 환경 설정
    setup_environment()
    
    # 여기에 추가 초기화 코드 작성
    
    logger.info("초기화 완료")
    
    # 여기에 애플리케이션 실행 코드 작성
    
    logger.info("Deep Research Chatbot 종료")

if __name__ == "__main__":
    main()