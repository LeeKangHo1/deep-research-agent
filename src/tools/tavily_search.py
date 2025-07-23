# src/tools/tavily_search.py

"""
Tavily API를 사용한 검색 도구 구현
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("tavily-python 패키지가 설치되어 있지 않습니다. 'pip install tavily-python'을 실행하세요.")

from src.models.search import SearchResult

# 로깅 설정
logger = logging.getLogger(__name__)

class TavilySearchException(Exception):
    """Tavily 검색 도구 관련 예외의 기본 클래스"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        self.timestamp = datetime.now()

class TavilyAPIKeyError(TavilySearchException):
    """API 키 관련 오류"""
    pass

class TavilyConnectionError(TavilySearchException):
    """Tavily API 연결 오류"""
    pass

class TavilyResponseError(TavilySearchException):
    """Tavily API 응답 오류"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.status_code = status_code
        self.response_body = response_body

class TavilyRateLimitError(TavilySearchException):
    """Tavily API 속도 제한 오류"""
    def __init__(self, message: str, retry_after: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.retry_after = retry_after

class TavilyTimeoutError(TavilySearchException):
    """Tavily API 타임아웃 오류"""
    pass

class TavilyAuthenticationError(TavilyAPIKeyError):
    """Tavily API 인증 오류"""
    pass

class TavilyQuotaExceededError(TavilySearchException):
    """Tavily API 할당량 초과 오류"""
    pass

class TavilySearchTool:
    """
    Tavily API를 사용하여 웹 검색을 수행하는 도구
    
    Attributes:
        api_key (str): Tavily API 키
        client (TavilyClient): Tavily API 클라이언트
    """
    
    def __init__(self):
        """
        TavilySearchTool 초기화
        
        API 키를 환경 변수에서 로드하고 클라이언트를 초기화합니다.
        """
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Optional[TavilyClient]:
        """
        Tavily API 클라이언트 초기화
        
        Returns:
            Optional[TavilyClient]: 초기화된 Tavily 클라이언트 또는 초기화 실패 시 None
            
        Raises:
            TavilyAPIKeyError: API 키가 없거나 유효하지 않은 경우
            TavilyConnectionError: API 서버에 연결할 수 없는 경우
            TavilyResponseError: API 서버에서 오류 응답을 반환한 경우
        """
        if not self.api_key:
            error_msg = "Tavily API 키가 설정되지 않았습니다. 환경 변수 TAVILY_API_KEY를 설정하세요."
            logger.error(error_msg)
            raise TavilyAPIKeyError(error_msg)
        
        # API 키 형식 검증 (기본적인 검증)
        if not self.api_key.startswith("tvly-"):
            error_msg = "Tavily API 키 형식이 올바르지 않습니다. 'tvly-'로 시작하는 API 키를 사용하세요."
            logger.error(error_msg)
            raise TavilyAPIKeyError(error_msg)
            
        try:
            client = TavilyClient(api_key=self.api_key)
            
            # API 키 유효성 검증을 위한 간단한 요청 시도
            # 이 부분은 실제 API 호출을 하지 않고 클라이언트만 생성하는 것으로 대체할 수 있습니다.
            # 실제 API 호출을 통한 검증은 첫 번째 검색 요청 시 수행됩니다.
            
            logger.info("Tavily API 클라이언트가 성공적으로 초기화되었습니다.")
            return client
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Tavily API 서버에 연결할 수 없습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilyConnectionError(error_msg) from e
            
        except requests.exceptions.Timeout as e:
            error_msg = f"Tavily API 서버 연결 시간이 초과되었습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilyConnectionError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Tavily API 요청 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilyConnectionError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Tavily API 클라이언트 초기화 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilySearchException(error_msg) from e
            
    @classmethod
    def from_env(cls) -> 'TavilySearchTool':
        """
        환경 변수에서 API 키를 로드하여 TavilySearchTool 인스턴스 생성
        
        Returns:
            TavilySearchTool: 생성된 TavilySearchTool 인스턴스
        """
        # 환경 변수가 로드되었는지 확인
        if not os.getenv("TAVILY_API_KEY"):
            # .env 파일이 있는지 확인하고 로드
            try:
                from dotenv import load_dotenv
                
                # 현재 디렉토리와 상위 디렉토리에서 .env 파일 찾기
                env_paths = ['.env', '../.env', '../../.env']
                for path in env_paths:
                    if os.path.exists(path):
                        load_dotenv(path)
                        logger.info(f".env 파일을 로드했습니다: {path}")
                        break
            except ImportError:
                logger.warning("python-dotenv 패키지가 설치되어 있지 않습니다. 'pip install python-dotenv'를 실행하세요.")
            except Exception as e:
                logger.error(f".env 파일 로드 중 오류 발생: {str(e)}")
        
        return cls()
        
    def is_api_key_valid(self) -> bool:
        """
        API 키가 유효한지 확인
        
        Returns:
            bool: API 키가 유효하면 True, 그렇지 않으면 False
        """
        return self.api_key is not None and self.client is not None
        
    @staticmethod
    def initialize_safely() -> 'TavilySearchTool':
        """
        예외 처리를 포함하여 안전하게 TavilySearchTool 인스턴스를 생성
        
        Returns:
            TavilySearchTool: 생성된 TavilySearchTool 인스턴스 또는 초기화 실패 시 제한된 기능의 인스턴스
        """
        try:
            return TavilySearchTool.from_env()
        except TavilyAPIKeyError as e:
            logger.warning(f"API 키 오류로 인해 제한된 기능으로 초기화합니다: {str(e)}")
            # API 키 없이 인스턴스 생성 (제한된 기능)
            instance = TavilySearchTool.__new__(TavilySearchTool)
            instance.api_key = None
            instance.client = None
            return instance
        except TavilyConnectionError as e:
            logger.warning(f"연결 오류로 인해 제한된 기능으로 초기화합니다: {str(e)}")
            # 연결 없이 인스턴스 생성 (제한된 기능)
            instance = TavilySearchTool.__new__(TavilySearchTool)
            instance.api_key = os.getenv("TAVILY_API_KEY")
            instance.client = None
            return instance
        except Exception as e:
            logger.error(f"TavilySearchTool 초기화 중 예상치 못한 오류가 발생했습니다: {str(e)}")
            # 기본 인스턴스 생성 (제한된 기능)
            instance = TavilySearchTool.__new__(TavilySearchTool)
            instance.api_key = None
            instance.client = None
            return instance
            
    def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> List[SearchResult]:
        """
        주어진 쿼리로 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            search_depth (str, optional): 검색 깊이 ("basic" 또는 "advanced"). 기본값은 "basic".
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            TavilyConnectionError: API 서버에 연결할 수 없는 경우
            TavilyResponseError: API 서버에서 오류 응답을 반환한 경우
            TavilySearchException: 기타 검색 관련 오류
        """
        if not self.client:
            error_msg = "Tavily API 클라이언트가 초기화되지 않았습니다."
            logger.error(error_msg)
            raise TavilySearchException(error_msg)
            
        # 검색 깊이 검증
        if search_depth not in ["basic", "advanced"]:
            logger.warning(f"유효하지 않은 검색 깊이: {search_depth}. 'basic'으로 설정합니다.")
            search_depth = "basic"
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        try:
            logger.info(f"Tavily 검색 수행: '{query}' (최대 결과 수: {max_results}, 검색 깊이: {search_depth})")
            
            # Tavily API를 사용하여 검색 수행
            # Tavily API는 max_results 파라미터를 직접 지원하지 않으므로 결과를 받은 후 제한
            response = self.client.search(query=query, search_depth=search_depth)
            
            # 응답 검증
            if not response or not isinstance(response, dict):
                error_msg = f"Tavily API에서 유효하지 않은 응답을 반환했습니다: {response}"
                logger.error(error_msg)
                raise TavilyResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(response)
            
            # 결과 수 제한
            if len(results) > max_results:
                logger.info(f"검색 결과를 {max_results}개로 제한합니다 (총 {len(results)}개).")
                results = results[:max_results]
            
            logger.info(f"Tavily 검색 완료: {len(results)}개의 결과를 찾았습니다.")
            return results
            
        except (requests.exceptions.RequestException, ValueError) as e:
            # API 오류 처리 메서드 호출
            self._handle_api_error(e, query)
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"Tavily 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilySearchException(error_msg, original_error=e)
            
    def search_with_context(self, query: str, context: str, max_results: int = 5, search_depth: str = "basic", 
                           context_weight: float = 0.5) -> List[SearchResult]:
        """
        컨텍스트를 포함한 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            context (str): 검색 컨텍스트 (이전 대화 또는 관련 정보)
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            search_depth (str, optional): 검색 깊이 ("basic" 또는 "advanced"). 기본값은 "basic".
            context_weight (float, optional): 컨텍스트 가중치 (0.0 ~ 1.0). 기본값은 0.5.
                0.0: 컨텍스트 무시, 1.0: 컨텍스트 최대 중요도
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            TavilyConnectionError: API 서버에 연결할 수 없는 경우
            TavilyResponseError: API 서버에서 오류 응답을 반환한 경우
            TavilySearchException: 기타 검색 관련 오류
        """
        if not self.client:
            error_msg = "Tavily API 클라이언트가 초기화되지 않았습니다."
            logger.error(error_msg)
            raise TavilySearchException(error_msg)
            
        # 검색 깊이 검증
        if search_depth not in ["basic", "advanced"]:
            logger.warning(f"유효하지 않은 검색 깊이: {search_depth}. 'basic'으로 설정합니다.")
            search_depth = "basic"
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 컨텍스트 가중치 검증
        if context_weight < 0.0 or context_weight > 1.0:
            logger.warning(f"유효하지 않은 컨텍스트 가중치: {context_weight}. 0.5로 설정합니다.")
            context_weight = 0.5
            
        try:
            # 컨텍스트와 쿼리를 결합하여 향상된 검색 쿼리 생성 (가중치 적용)
            enhanced_query = self._enhance_query_with_context(query, context, context_weight)
            
            logger.info(f"Tavily 컨텍스트 기반 검색 수행: '{enhanced_query}' (최대 결과 수: {max_results}, 검색 깊이: {search_depth}, 컨텍스트 가중치: {context_weight})")
            
            # Tavily API를 사용하여 검색 수행
            response = self.client.search(query=enhanced_query, search_depth=search_depth)
            
            # 응답 검증
            if not response or not isinstance(response, dict):
                error_msg = f"Tavily API에서 유효하지 않은 응답을 반환했습니다: {response}"
                logger.error(error_msg)
                raise TavilyResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(response)
            
            # 결과 수 제한
            if len(results) > max_results:
                logger.info(f"검색 결과를 {max_results}개로 제한합니다 (총 {len(results)}개).")
                results = results[:max_results]
            
            logger.info(f"Tavily 컨텍스트 기반 검색 완료: {len(results)}개의 결과를 찾았습니다.")
            return results
            
        except (requests.exceptions.RequestException, ValueError) as e:
            # API 오류 처리 메서드 호출
            self._handle_api_error(e, enhanced_query if 'enhanced_query' in locals() else query)
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"Tavily 컨텍스트 기반 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise TavilySearchException(error_msg, original_error=e)
            
    def _enhance_query_with_context(self, query: str, context: str, context_weight: float = 0.5) -> str:
        """
        컨텍스트를 사용하여 검색 쿼리를 향상
        
        Args:
            query (str): 원본 검색 쿼리
            context (str): 검색 컨텍스트
            context_weight (float, optional): 컨텍스트 가중치 (0.0 ~ 1.0). 기본값은 0.5.
            
        Returns:
            str: 향상된 검색 쿼리
        """
        # 컨텍스트 가중치가 0이면 원본 쿼리 반환
        if context_weight <= 0.0:
            logger.info("컨텍스트 가중치가 0이므로 원본 쿼리를 사용합니다.")
            return query
            
        # 컨텍스트가 없거나 너무 짧은 경우 원본 쿼리 반환
        if not context or len(context.strip()) < 10:
            return query
            
        # 컨텍스트가 너무 긴 경우 잘라내기 (Tavily API 제한 고려)
        max_context_length = 500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        # 컨텍스트에서 중요한 키워드 추출
        keywords = self._extract_keywords_from_context(context)
        
        # 키워드가 없는 경우 원본 쿼리 반환
        if not keywords:
            return query
            
        # 컨텍스트 가중치에 따라 키워드 수 조정
        if context_weight < 1.0:
            # 가중치에 따라 키워드 수 결정
            keyword_list = keywords.split()
            num_keywords = max(1, int(len(keyword_list) * context_weight))
            keywords = " ".join(keyword_list[:num_keywords])
            
        # 키워드가 있는 경우 쿼리에 추가
        enhanced_query = f"{query} {keywords}"
        logger.info(f"쿼리가 향상되었습니다 (가중치 {context_weight}): '{query}' -> '{enhanced_query}'")
        return enhanced_query
        
    def _extract_keywords_from_context(self, context: str) -> str:
        """
        컨텍스트에서 중요한 키워드 추출 (간단한 구현)
        
        Args:
            context (str): 검색 컨텍스트
            
        Returns:
            str: 추출된 키워드 문자열
        """
        # 간단한 구현: 불용어 제거 및 빈도 기반 키워드 추출
        # 실제 구현에서는 NLP 라이브러리를 사용하여 더 정교한 키워드 추출 가능
        
        # 불용어 목록 (간단한 예시)
        stopwords = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "who", "which", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "to", "at", "in", "on", "for", "with", "by", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "of", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "should", "now"
        ])
        
        # 한국어 불용어 추가
        korean_stopwords = set([
            "이", "그", "저", "것", "이것", "그것", "저것", "이번", "그번", "저번",
            "이런", "그런", "저런", "하다", "있다", "되다", "없다", "나", "너", "우리",
            "저희", "당신", "그들", "그녀", "그", "이", "저", "그것", "이것", "저것",
            "그런", "이런", "저런", "어떤", "무슨", "어느", "몇", "언제", "어디", "어떻게",
            "왜", "무엇", "누구", "어디", "언제", "어떻게", "왜", "이렇게", "그렇게", "저렇게",
            "이와", "그와", "저와", "에", "에서", "의", "을", "를", "이", "가", "은", "는",
            "로", "으로", "와", "과", "도", "만", "까지", "부터", "에게", "께", "처럼", "같이"
        ])
        
        # 불용어 통합
        all_stopwords = stopwords.union(korean_stopwords)
        
        # 텍스트 전처리
        # 소문자 변환 및 구두점 제거
        text = context.lower()
        for punct in ".,;:!?\"'()[]{}":
            text = text.replace(punct, " ")
            
        # 단어 분리 및 불용어 제거
        words = [word for word in text.split() if word not in all_stopwords and len(word) > 1]
        
        # 단어 빈도 계산
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # 빈도 기준으로 상위 5개 키워드 선택
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 키워드 문자열 생성
        keywords = " ".join([word for word, _ in top_keywords])
        
        return keywords
        
    def _parse_search_results(self, response: Dict[str, Any]) -> List[SearchResult]:
        """
        Tavily API 응답을 SearchResult 객체 목록으로 파싱
        
        Args:
            response (Dict[str, Any]): Tavily API 응답
            
        Returns:
            List[SearchResult]: 파싱된 검색 결과 목록
        """
        results = []
        
        # 응답에서 결과 목록 추출
        raw_results = response.get("results", [])
        
        # 응답에서 추가 메타데이터 추출
        query = response.get("query", "")
        search_depth = response.get("search_depth", "basic")
        response_time = response.get("response_time", 0)
        
        # 결과가 없는 경우 로그 기록
        if not raw_results:
            logger.warning(f"검색 쿼리 '{query}'에 대한 결과가 없습니다.")
            return []
            
        for index, raw_result in enumerate(raw_results):
            try:
                # 필수 필드 확인
                title = raw_result.get("title", "제목 없음")
                url = raw_result.get("url", "")
                content = raw_result.get("content", "")
                
                # URL이 없는 경우 건너뛰기
                if not url:
                    logger.warning(f"URL이 없는 검색 결과를 건너뜁니다: {title}")
                    continue
                    
                # 내용이 없는 경우 스니펫을 내용으로 사용
                if not content and raw_result.get("snippet"):
                    content = raw_result.get("snippet")
                    logger.info(f"내용이 없어 스니펫을 내용으로 사용합니다: {title}")
                
                # 내용과 제목이 모두 없는 경우 건너뛰기
                if not content and not title:
                    logger.warning("내용과 제목이 모두 없는 검색 결과를 건너뜁니다.")
                    continue
                
                # 점수 계산 (Tavily는 기본 점수를 제공하지 않으므로 순위 기반 점수 계산)
                # 첫 번째 결과가 가장 높은 점수를 가짐
                score = 1.0 - (index * 0.05)  # 각 결과마다 0.05씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
                
                # 발행 날짜 파싱 (있는 경우)
                published_date = None
                if "published_date" in raw_result:
                    try:
                        # ISO 형식 날짜 파싱 시도
                        published_date = datetime.fromisoformat(raw_result["published_date"].replace('Z', '+00:00'))
                    except (ValueError, TypeError, AttributeError):
                        try:
                            # 다양한 날짜 형식 파싱 시도
                            from dateutil import parser
                            published_date = parser.parse(str(raw_result["published_date"]))
                        except (ImportError, ValueError, TypeError):
                            logger.warning(f"발행 날짜를 파싱할 수 없습니다: {raw_result.get('published_date')}")
                
                # 소스 유형 결정
                source_type = raw_result.get("source_type", "webpage")
                if not source_type:
                    # URL 기반으로 소스 유형 추정
                    url_lower = url.lower()
                    if url_lower.endswith(".pdf"):
                        source_type = "pdf"
                    elif url_lower.endswith((".doc", ".docx")):
                        source_type = "document"
                    elif url_lower.endswith((".ppt", ".pptx")):
                        source_type = "presentation"
                    elif url_lower.endswith((".xls", ".xlsx")):
                        source_type = "spreadsheet"
                    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
                        source_type = "video"
                    elif "github.com" in url_lower:
                        source_type = "code"
                    else:
                        source_type = "webpage"
                
                # 도메인 추출
                domain = ""
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                except (ImportError, ValueError, AttributeError):
                    # URL에서 도메인 추출 실패 시 간단한 방법 사용
                    domain_parts = url.split("//")
                    if len(domain_parts) > 1:
                        domain = domain_parts[1].split("/")[0]
                
                # 스니펫 정리 (HTML 태그 제거)
                snippet = raw_result.get("snippet", "")
                if snippet:
                    # 간단한 HTML 태그 제거
                    for tag in ["<p>", "</p>", "<br>", "<br/>", "<br />", "<div>", "</div>"]:
                        snippet = snippet.replace(tag, " ")
                    # 연속된 공백 제거
                    import re
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                
                # 메타데이터 설정
                metadata = {
                    "search_engine": "tavily",
                    "domain": domain,
                    "rank": index + 1,  # 검색 결과 순위
                    "query": query,
                    "search_depth": search_depth,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "raw_data": raw_result
                }
                
                # 추가 메타데이터 포함 (있는 경우)
                for key in ["author", "category", "language", "favicon"]:
                    if key in raw_result:
                        metadata[key] = raw_result[key]
                
                # SearchResult 객체 생성
                result = SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=score,
                    published_date=published_date,
                    snippet=snippet,
                    source_type=source_type,
                    metadata=metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"검색 결과 파싱 중 오류 발생: {str(e)}")
                continue
        
        # 결과 정렬 (점수 기준 내림차순)
        results.sort(key=lambda x: x.score, reverse=True)
                
        return results
        
    @staticmethod
    def convert_to_search_result(raw_result: Dict[str, Any], query: str = "", rank: int = 0) -> Optional[SearchResult]:
        """
        단일 Tavily API 결과를 SearchResult 객체로 변환
        
        Args:
            raw_result (Dict[str, Any]): Tavily API 결과 항목
            query (str, optional): 검색 쿼리. 기본값은 빈 문자열.
            rank (int, optional): 검색 결과 순위. 기본값은 0.
            
        Returns:
            Optional[SearchResult]: 변환된 SearchResult 객체 또는 변환 실패 시 None
        """
        try:
            # 필수 필드 확인
            title = raw_result.get("title", "제목 없음")
            url = raw_result.get("url", "")
            content = raw_result.get("content", "")
            
            # URL이 없는 경우 None 반환
            if not url:
                logger.warning(f"URL이 없는 검색 결과를 변환할 수 없습니다: {title}")
                return None
                
            # 내용이 없는 경우 스니펫을 내용으로 사용
            if not content and raw_result.get("snippet"):
                content = raw_result.get("snippet")
            
            # 내용과 제목이 모두 없는 경우 None 반환
            if not content and not title:
                logger.warning("내용과 제목이 모두 없는 검색 결과를 변환할 수 없습니다.")
                return None
            
            # 점수 계산 (순위 기반)
            score = 1.0
            if rank > 0:
                score = 1.0 - (rank * 0.05)  # 각 결과마다 0.05씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
            
            # 발행 날짜 파싱 (있는 경우)
            published_date = None
            if "published_date" in raw_result:
                try:
                    # ISO 형식 날짜 파싱 시도
                    published_date = datetime.fromisoformat(raw_result["published_date"].replace('Z', '+00:00'))
                except (ValueError, TypeError, AttributeError):
                    try:
                        # 다양한 날짜 형식 파싱 시도
                        from dateutil import parser
                        published_date = parser.parse(str(raw_result["published_date"]))
                    except (ImportError, ValueError, TypeError):
                        logger.warning(f"발행 날짜를 파싱할 수 없습니다: {raw_result.get('published_date')}")
            
            # 소스 유형 결정
            source_type = raw_result.get("source_type", "webpage")
            if not source_type:
                # URL 기반으로 소스 유형 추정
                url_lower = url.lower()
                if url_lower.endswith(".pdf"):
                    source_type = "pdf"
                elif url_lower.endswith((".doc", ".docx")):
                    source_type = "document"
                elif url_lower.endswith((".ppt", ".pptx")):
                    source_type = "presentation"
                elif url_lower.endswith((".xls", ".xlsx")):
                    source_type = "spreadsheet"
                elif "youtube.com" in url_lower or "youtu.be" in url_lower:
                    source_type = "video"
                elif "github.com" in url_lower:
                    source_type = "code"
                else:
                    source_type = "webpage"
            
            # 도메인 추출
            domain = ""
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
            except (ImportError, ValueError, AttributeError):
                # URL에서 도메인 추출 실패 시 간단한 방법 사용
                domain_parts = url.split("//")
                if len(domain_parts) > 1:
                    domain = domain_parts[1].split("/")[0]
            
            # 스니펫 정리 (HTML 태그 제거)
            snippet = raw_result.get("snippet", "")
            if snippet:
                # 간단한 HTML 태그 제거
                for tag in ["<p>", "</p>", "<br>", "<br/>", "<br />", "<div>", "</div>"]:
                    snippet = snippet.replace(tag, " ")
                # 연속된 공백 제거
                import re
                snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            # 메타데이터 설정
            metadata = {
                "search_engine": "tavily",
                "domain": domain,
                "rank": rank,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "raw_data": raw_result
            }
            
            # 추가 메타데이터 포함 (있는 경우)
            for key in ["author", "category", "language", "favicon"]:
                if key in raw_result:
                    metadata[key] = raw_result[key]
            
            # SearchResult 객체 생성
            result = SearchResult(
                title=title,
                url=url,
                content=content,
                score=score,
                published_date=published_date,
                snippet=snippet,
                source_type=source_type,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"SearchResult 객체 변환 중 오류 발생: {str(e)}")
            return None
            
    @classmethod
    def convert_raw_results(cls, raw_results: List[Dict[str, Any]], query: str = "") -> List[SearchResult]:
        """
        Tavily API 결과 목록을 SearchResult 객체 목록으로 변환
        
        Args:
            raw_results (List[Dict[str, Any]]): Tavily API 결과 목록
            query (str, optional): 검색 쿼리. 기본값은 빈 문자열.
            
        Returns:
            List[SearchResult]: 변환된 SearchResult 객체 목록
        """
        results = []
        
        for index, raw_result in enumerate(raw_results):
            result = cls.convert_to_search_result(raw_result, query, index + 1)
            if result:
                results.append(result)
                
        return results
        
    def _handle_api_error(self, error: Exception, query: str) -> None:
        """
        Tavily API 오류 처리 및 적절한 예외 발생
        
        Args:
            error (Exception): 원본 예외
            query (str): 검색 쿼리
            
        Raises:
            TavilyAuthenticationError: 인증 오류 발생 시
            TavilyRateLimitError: 속도 제한 오류 발생 시
            TavilyQuotaExceededError: 할당량 초과 오류 발생 시
            TavilyTimeoutError: 타임아웃 오류 발생 시
            TavilyConnectionError: 연결 오류 발생 시
            TavilyResponseError: 응답 오류 발생 시
            TavilySearchException: 기타 오류 발생 시
        """
        error_message = str(error)
        
        # requests 예외 처리
        if isinstance(error, requests.exceptions.RequestException):
            # 응답이 있는 경우 상태 코드 및 응답 본문 추출
            response = getattr(error, "response", None)
            status_code = None
            response_body = None
            
            if response:
                status_code = response.status_code
                try:
                    response_body = response.text
                except:
                    response_body = "응답 본문을 읽을 수 없습니다."
                    
                # 상태 코드 기반 오류 처리
                if status_code == 401:
                    logger.error(f"Tavily API 인증 오류 (쿼리: '{query}'): {error_message}")
                    raise TavilyAuthenticationError(f"Tavily API 인증 오류: {error_message}", original_error=error)
                    
                elif status_code == 429:
                    # 재시도 시간 추출
                    retry_after = None
                    try:
                        retry_after = int(response.headers.get("Retry-After", 60))
                    except (ValueError, TypeError):
                        retry_after = 60
                        
                    logger.error(f"Tavily API 속도 제한 오류 (쿼리: '{query}'): {error_message}, {retry_after}초 후 재시도")
                    raise TavilyRateLimitError(f"Tavily API 속도 제한 오류: {error_message}", retry_after=retry_after, original_error=error)
                    
                elif status_code == 402:
                    logger.error(f"Tavily API 할당량 초과 오류 (쿼리: '{query}'): {error_message}")
                    raise TavilyQuotaExceededError(f"Tavily API 할당량 초과 오류: {error_message}", original_error=error)
                    
                else:
                    logger.error(f"Tavily API 응답 오류 (쿼리: '{query}', 상태 코드: {status_code}): {error_message}")
                    raise TavilyResponseError(f"Tavily API 응답 오류 (상태 코드: {status_code}): {error_message}", 
                                             status_code=status_code, response_body=response_body, original_error=error)
            
            # 타임아웃 오류 처리
            if isinstance(error, requests.exceptions.Timeout):
                logger.error(f"Tavily API 타임아웃 오류 (쿼리: '{query}'): {error_message}")
                raise TavilyTimeoutError(f"Tavily API 타임아웃 오류: {error_message}", original_error=error)
                
            # 연결 오류 처리
            if isinstance(error, requests.exceptions.ConnectionError):
                logger.error(f"Tavily API 연결 오류 (쿼리: '{query}'): {error_message}")
                raise TavilyConnectionError(f"Tavily API 연결 오류: {error_message}", original_error=error)
                
            # 기타 요청 오류 처리
            logger.error(f"Tavily API 요청 오류 (쿼리: '{query}'): {error_message}")
            raise TavilyConnectionError(f"Tavily API 요청 오류: {error_message}", original_error=error)
            
        # JSON 파싱 오류 처리
        if isinstance(error, ValueError) and "JSON" in error_message:
            logger.error(f"Tavily API 응답 파싱 오류 (쿼리: '{query}'): {error_message}")
            raise TavilyResponseError(f"Tavily API 응답 파싱 오류: {error_message}", original_error=error)
            
        # 기타 오류 처리
        logger.error(f"Tavily 검색 중 예상치 못한 오류 발생 (쿼리: '{query}'): {error_message}")
        raise TavilySearchException(f"Tavily 검색 중 예상치 못한 오류 발생: {error_message}", original_error=error)
        
    def search_with_retry(self, query: str, max_results: int = 5, search_depth: str = "basic", 
                        max_retries: int = 3, retry_delay: float = 1.0, 
                        backoff_factor: float = 2.0) -> List[SearchResult]:
        """
        재시도 메커니즘을 포함한 웹 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            search_depth (str, optional): 검색 깊이 ("basic" 또는 "advanced"). 기본값은 "basic".
            max_retries (int, optional): 최대 재시도 횟수. 기본값은 3.
            retry_delay (float, optional): 초기 재시도 지연 시간(초). 기본값은 1.0.
            backoff_factor (float, optional): 지수 백오프 계수. 기본값은 2.0.
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            TavilyConnectionError: 모든 재시도 후에도 API 서버에 연결할 수 없는 경우
            TavilyResponseError: 모든 재시도 후에도 API 서버에서 오류 응답을 반환한 경우
            TavilySearchException: 기타 검색 관련 오류
        """
        if not self.client:
            error_msg = "Tavily API 클라이언트가 초기화되지 않았습니다."
            logger.error(error_msg)
            raise TavilySearchException(error_msg)
            
        # 검색 깊이 검증
        if search_depth not in ["basic", "advanced"]:
            logger.warning(f"유효하지 않은 검색 깊이: {search_depth}. 'basic'으로 설정합니다.")
            search_depth = "basic"
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 재시도 파라미터 검증
        if max_retries < 0:
            logger.warning(f"유효하지 않은 최대 재시도 횟수: {max_retries}. 3으로 설정합니다.")
            max_retries = 3
            
        if retry_delay < 0:
            logger.warning(f"유효하지 않은 재시도 지연 시간: {retry_delay}. 1.0으로 설정합니다.")
            retry_delay = 1.0
            
        if backoff_factor < 1.0:
            logger.warning(f"유효하지 않은 백오프 계수: {backoff_factor}. 2.0으로 설정합니다.")
            backoff_factor = 2.0
            
        # 재시도 가능한 오류 유형
        retryable_errors = (
            TavilyConnectionError,
            TavilyTimeoutError,
            TavilyRateLimitError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout
        )
        
        # 재시도 불가능한 오류 유형
        non_retryable_errors = (
            TavilyAPIKeyError,
            TavilyAuthenticationError,
            TavilyQuotaExceededError
        )
        
        last_exception = None
        current_retry = 0
        current_delay = retry_delay
        
        while current_retry <= max_retries:
            try:
                # 첫 번째 시도가 아닌 경우 로그 기록
                if current_retry > 0:
                    logger.info(f"Tavily 검색 재시도 {current_retry}/{max_retries}: '{query}'")
                
                # 검색 수행
                return self.search(query=query, max_results=max_results, search_depth=search_depth)
                
            except non_retryable_errors as e:
                # 재시도 불가능한 오류는 즉시 다시 발생
                logger.error(f"재시도 불가능한 오류 발생: {str(e)}")
                raise
                
            except retryable_errors as e:
                # 재시도 가능한 오류 처리
                last_exception = e
                
                # 속도 제한 오류인 경우 지정된 시간 동안 대기
                if isinstance(e, TavilyRateLimitError) and e.retry_after:
                    current_delay = e.retry_after
                    logger.warning(f"속도 제한 오류로 인해 {current_delay}초 동안 대기 후 재시도합니다.")
                else:
                    logger.warning(f"재시도 가능한 오류 발생: {str(e)}. {current_delay:.1f}초 후 재시도합니다.")
                
                # 마지막 재시도인 경우 예외 발생
                if current_retry >= max_retries:
                    logger.error(f"최대 재시도 횟수({max_retries})에 도달했습니다. 마지막 오류: {str(e)}")
                    raise last_exception
                
                # 지연 후 재시도
                time.sleep(current_delay)
                current_retry += 1
                current_delay *= backoff_factor
                
            except Exception as e:
                # 기타 예상치 못한 오류
                logger.error(f"예상치 못한 오류 발생: {str(e)}")
                raise TavilySearchException(f"Tavily 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}", original_error=e)
                
        # 모든 재시도 실패 시 마지막 예외 발생
        if last_exception:
            raise last_exception
            
        # 이 코드에 도달하면 안 됨
        raise TavilySearchException("알 수 없는 오류로 인해 검색에 실패했습니다.")
        
    def search_with_context_and_retry(self, query: str, context: str, max_results: int = 5, 
                                     search_depth: str = "basic", context_weight: float = 0.5,
                                     max_retries: int = 3, retry_delay: float = 1.0, 
                                     backoff_factor: float = 2.0) -> List[SearchResult]:
        """
        재시도 메커니즘을 포함한 컨텍스트 기반 웹 검색 수행
        
        Args:
            query (str): 검색 쿼리
            context (str): 검색 컨텍스트
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            search_depth (str, optional): 검색 깊이 ("basic" 또는 "advanced"). 기본값은 "basic".
            context_weight (float, optional): 컨텍스트 가중치 (0.0 ~ 1.0). 기본값은 0.5.
            max_retries (int, optional): 최대 재시도 횟수. 기본값은 3.
            retry_delay (float, optional): 초기 재시도 지연 시간(초). 기본값은 1.0.
            backoff_factor (float, optional): 지수 백오프 계수. 기본값은 2.0.
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            TavilyConnectionError: 모든 재시도 후에도 API 서버에 연결할 수 없는 경우
            TavilyResponseError: 모든 재시도 후에도 API 서버에서 오류 응답을 반환한 경우
            TavilySearchException: 기타 검색 관련 오류
        """
        if not self.client:
            error_msg = "Tavily API 클라이언트가 초기화되지 않았습니다."
            logger.error(error_msg)
            raise TavilySearchException(error_msg)
            
        # 검색 깊이 검증
        if search_depth not in ["basic", "advanced"]:
            logger.warning(f"유효하지 않은 검색 깊이: {search_depth}. 'basic'으로 설정합니다.")
            search_depth = "basic"
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 컨텍스트 가중치 검증
        if context_weight < 0.0 or context_weight > 1.0:
            logger.warning(f"유효하지 않은 컨텍스트 가중치: {context_weight}. 0.5로 설정합니다.")
            context_weight = 0.5
            
        # 재시도 파라미터 검증
        if max_retries < 0:
            logger.warning(f"유효하지 않은 최대 재시도 횟수: {max_retries}. 3으로 설정합니다.")
            max_retries = 3
            
        if retry_delay < 0:
            logger.warning(f"유효하지 않은 재시도 지연 시간: {retry_delay}. 1.0으로 설정합니다.")
            retry_delay = 1.0
            
        if backoff_factor < 1.0:
            logger.warning(f"유효하지 않은 백오프 계수: {backoff_factor}. 2.0으로 설정합니다.")
            backoff_factor = 2.0
            
        # 컨텍스트와 쿼리를 결합하여 향상된 검색 쿼리 생성 (가중치 적용)
        enhanced_query = self._enhance_query_with_context(query, context, context_weight)
            
        # 재시도 가능한 오류 유형
        retryable_errors = (
            TavilyConnectionError,
            TavilyTimeoutError,
            TavilyRateLimitError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout
        )
        
        # 재시도 불가능한 오류 유형
        non_retryable_errors = (
            TavilyAPIKeyError,
            TavilyAuthenticationError,
            TavilyQuotaExceededError
        )
        
        last_exception = None
        current_retry = 0
        current_delay = retry_delay
        
        while current_retry <= max_retries:
            try:
                # 첫 번째 시도가 아닌 경우 로그 기록
                if current_retry > 0:
                    logger.info(f"Tavily 컨텍스트 기반 검색 재시도 {current_retry}/{max_retries}: '{enhanced_query}'")
                
                # 검색 수행
                return self.search_with_context(
                    query=query, 
                    context=context, 
                    max_results=max_results, 
                    search_depth=search_depth,
                    context_weight=context_weight
                )
                
            except non_retryable_errors as e:
                # 재시도 불가능한 오류는 즉시 다시 발생
                logger.error(f"재시도 불가능한 오류 발생: {str(e)}")
                raise
                
            except retryable_errors as e:
                # 재시도 가능한 오류 처리
                last_exception = e
                
                # 속도 제한 오류인 경우 지정된 시간 동안 대기
                if isinstance(e, TavilyRateLimitError) and e.retry_after:
                    current_delay = e.retry_after
                    logger.warning(f"속도 제한 오류로 인해 {current_delay}초 동안 대기 후 재시도합니다.")
                else:
                    logger.warning(f"재시도 가능한 오류 발생: {str(e)}. {current_delay:.1f}초 후 재시도합니다.")
                
                # 마지막 재시도인 경우 예외 발생
                if current_retry >= max_retries:
                    logger.error(f"최대 재시도 횟수({max_retries})에 도달했습니다. 마지막 오류: {str(e)}")
                    raise last_exception
                
                # 지연 후 재시도
                time.sleep(current_delay)
                current_retry += 1
                current_delay *= backoff_factor
                
            except Exception as e:
                # 기타 예상치 못한 오류
                logger.error(f"예상치 못한 오류 발생: {str(e)}")
                raise TavilySearchException(f"Tavily 컨텍스트 기반 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}", original_error=e)
                
        # 모든 재시도 실패 시 마지막 예외 발생
        if last_exception:
            raise last_exception
            
        # 이 코드에 도달하면 안 됨
        raise TavilySearchException("알 수 없는 오류로 인해 검색에 실패했습니다.")