# src/utils/error_utils.py

"""
오류 처리 유틸리티 모듈
애플리케이션 오류 처리 및 관리를 위한 유틸리티 함수를 제공합니다.
"""

from typing import Any, Dict, Optional, Type, Union
from .logging_utils import logger

class ChatbotError(Exception):
    """챗봇 기본 예외 클래스"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class APIError(ChatbotError):
    """외부 API 호출 관련 예외"""
    pass

class ConfigError(ChatbotError):
    """설정 관련 예외"""
    pass

class AgentError(ChatbotError):
    """에이전트 관련 예외"""
    pass

class WorkflowError(ChatbotError):
    """워크플로우 관련 예외"""
    pass

class ValidationError(ChatbotError):
    """데이터 검증 관련 예외"""
    pass

def handle_error(
    error: Union[Exception, str],
    error_type: Type[ChatbotError] = ChatbotError,
    details: Optional[Dict[str, Any]] = None,
    log_level: str = "ERROR"
) -> ChatbotError:
    """
    예외를 처리하고 로깅합니다.
    
    Args:
        error: 예외 객체 또는 오류 메시지
        error_type: 반환할 예외 타입
        details: 추가 오류 세부 정보
        log_level: 로그 레벨
        
    Returns:
        처리된 예외 객체
    """
    # 오류 메시지 추출
    if isinstance(error, Exception):
        message = str(error)
    else:
        message = error
    
    # 세부 정보 설정
    error_details = details or {}
    
    # 로깅
    log_method = getattr(logger, log_level.lower(), logger.error)
    log_method(f"{message} - {error_details}")
    
    # 예외 객체 생성 및 반환
    return error_type(message, error_details)

def graceful_failure(
    error: Union[Exception, str],
    fallback_value: Any,
    error_type: Type[ChatbotError] = ChatbotError,
    details: Optional[Dict[str, Any]] = None,
    log_level: str = "WARNING"
) -> Any:
    """
    예외를 처리하고 대체 값을 반환합니다.
    
    Args:
        error: 예외 객체 또는 오류 메시지
        fallback_value: 오류 발생 시 반환할 대체 값
        error_type: 로깅할 예외 타입
        details: 추가 오류 세부 정보
        log_level: 로그 레벨
        
    Returns:
        대체 값
    """
    handle_error(error, error_type, details, log_level)
    return fallback_value