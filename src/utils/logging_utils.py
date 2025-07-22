# src/utils/logging_utils.py

"""
로깅 유틸리티 모듈
애플리케이션 로깅 설정 및 관리를 위한 유틸리티 함수를 제공합니다.
"""

import logging
import sys
from typing import Optional
from ..config import Config

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    로거를 설정하고 반환합니다.
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (기본값: Config.LOG_LEVEL)
        
    Returns:
        설정된 로거 인스턴스
    """
    if level is None:
        level = Config.LOG_LEVEL
        
    logger = logging.getLogger(name)
    
    # 로그 레벨 설정
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 이미 핸들러가 설정되어 있으면 추가하지 않음
    if not logger.handlers:
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(console_handler)
    
    return logger

# 기본 로거 생성
logger = setup_logger("deep_research_chatbot")