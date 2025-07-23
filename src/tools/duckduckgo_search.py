# src/tools/duckduckgo_search.py

"""
DuckDuckGo API를 사용한 검색 도구 구현
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("duckduckgo_search 패키지가 설치되어 있지 않습니다. 'pip install duckduckgo-search'를 실행하세요.")

from src.models.search import SearchResult

# 로깅 설정
logger = logging.getLogger(__name__)

class DuckDuckGoSearchException(Exception):
    """DuckDuckGo 검색 도구 관련 예외의 기본 클래스"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        self.timestamp = datetime.now()

class DuckDuckGoConnectionError(DuckDuckGoSearchException):
    """DuckDuckGo API 연결 오류"""
    pass

class DuckDuckGoResponseError(DuckDuckGoSearchException):
    """DuckDuckGo API 응답 오류"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.status_code = status_code
        self.response_body = response_body

class DuckDuckGoRateLimitError(DuckDuckGoSearchException):
    """DuckDuckGo API 속도 제한 오류"""
    def __init__(self, message: str, retry_after: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.retry_after = retry_after

class DuckDuckGoTimeoutError(DuckDuckGoSearchException):
    """DuckDuckGo API 타임아웃 오류"""
    pass

class DuckDuckGoSearchTool:
    """
    DuckDuckGo API를 사용하여 웹 검색을 수행하는 도구
    
    Attributes:
        client (DDGS): DuckDuckGo 검색 클라이언트
    """
    
    def __init__(self):
        """
        DuckDuckGoSearchTool 초기화
        
        클라이언트를 초기화합니다.
        """
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Optional[DDGS]:
        """
        DuckDuckGo 검색 클라이언트 초기화
        
        Returns:
            Optional[DDGS]: 초기화된 DuckDuckGo 클라이언트 또는 초기화 실패 시 None
            
        Raises:
            DuckDuckGoConnectionError: API 서버에 연결할 수 없는 경우
            DuckDuckGoSearchException: 기타 초기화 관련 오류
        """
        try:
            # 프록시 설정 확인 (환경 변수에서)
            proxies = None
            http_proxy = os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("HTTPS_PROXY")
            
            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies["http"] = http_proxy
                if https_proxy:
                    proxies["https"] = https_proxy
                logger.info(f"프록시 설정이 감지되었습니다: {proxies}")
            
            # 타임아웃 설정 (환경 변수에서, 기본값 30초)
            timeout = int(os.getenv("DUCKDUCKGO_TIMEOUT", "30"))
            
            # DuckDuckGo 클라이언트 초기화
            # DDGS는 프록시와 타임아웃을 지원하지만, 여기서는 기본 설정으로 초기화
            client = DDGS()
            
            # 클라이언트 연결 테스트
            self._test_client_connection(client)
            
            logger.info("DuckDuckGo 검색 클라이언트가 성공적으로 초기화되었습니다.")
            return client
            
        except DuckDuckGoConnectionError as e:
            # 이미 적절한 예외가 발생했으므로 그대로 전파
            raise e
            
        except Exception as e:
            error_msg = f"DuckDuckGo 검색 클라이언트 초기화 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg) from e
            
    def _test_client_connection(self, client: DDGS) -> None:
        """
        DuckDuckGo 클라이언트 연결 테스트
        
        Args:
            client (DDGS): 테스트할 DuckDuckGo 클라이언트
            
        Raises:
            DuckDuckGoConnectionError: 연결 테스트 실패 시
        """
        try:
            # 간단한 검색으로 연결 테스트 (첫 번째 결과만 가져오기)
            # 실제 API 호출을 최소화하기 위해 제한된 결과만 요청
            test_query = "test connection"
            test_results = list(client.text(test_query, max_results=1))
            
            # 결과가 없어도 연결은 성공한 것으로 간주
            logger.info("DuckDuckGo 클라이언트 연결 테스트 성공")
            
        except Exception as e:
            error_msg = f"DuckDuckGo 클라이언트 연결 테스트 실패: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoConnectionError(error_msg, original_error=e)
            
    @classmethod
    def initialize_safely(cls) -> 'DuckDuckGoSearchTool':
        """
        예외 처리를 포함하여 안전하게 DuckDuckGoSearchTool 인스턴스를 생성
        
        Returns:
            DuckDuckGoSearchTool: 생성된 DuckDuckGoSearchTool 인스턴스 또는 초기화 실패 시 제한된 기능의 인스턴스
        """
        try:
            return cls()
        except DuckDuckGoConnectionError as e:
            logger.warning(f"연결 오류로 인해 제한된 기능으로 초기화합니다: {str(e)}")
            # 연결 없이 인스턴스 생성 (제한된 기능)
            instance = cls.__new__(cls)
            instance.client = None
            return instance
        except Exception as e:
            logger.error(f"DuckDuckGoSearchTool 초기화 중 예상치 못한 오류가 발생했습니다: {str(e)}")
            # 기본 인스턴스 생성 (제한된 기능)
            instance = cls.__new__(cls)
            instance.client = None
            return instance
            
    def is_client_available(self) -> bool:
        """
        DuckDuckGo 클라이언트가 사용 가능한지 확인
        
        Returns:
            bool: 클라이언트가 사용 가능하면 True, 그렇지 않으면 False
        """
        return self.client is not None
        
    def ensure_client_available(self) -> None:
        """
        클라이언트가 사용 가능한지 확인하고, 사용할 수 없으면 예외 발생
        
        Raises:
            DuckDuckGoSearchException: 클라이언트를 사용할 수 없는 경우
        """
        if not self.is_client_available():
            error_msg = "DuckDuckGo 검색 클라이언트를 사용할 수 없습니다. 초기화에 실패했거나 연결 문제가 있을 수 있습니다."
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg)
            
    def search(self, query: str, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate") -> List[SearchResult]:
        """
        주어진 쿼리로 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            DuckDuckGoConnectionError: API 서버에 연결할 수 없는 경우
            DuckDuckGoResponseError: API 서버에서 오류 응답을 반환한 경우
            DuckDuckGoSearchException: 기타 검색 관련 오류
        """
        # 클라이언트 사용 가능 여부 확인
        self.ensure_client_available()
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 세이프서치 설정 검증
        if safesearch not in ["on", "moderate", "off"]:
            logger.warning(f"유효하지 않은 세이프서치 설정: {safesearch}. 'moderate'로 설정합니다.")
            safesearch = "moderate"
            
        try:
            logger.info(f"DuckDuckGo 검색 수행: '{query}' (최대 결과 수: {max_results}, 지역: {region}, 세이프서치: {safesearch})")
            
            # DuckDuckGo API를 사용하여 검색 수행
            # DDGS.text() 메서드는 제너레이터를 반환하므로 list()로 변환
            raw_results = list(self.client.text(
                query, 
                region=region, 
                safesearch=safesearch, 
                max_results=max_results
            ))
            
            # 응답 검증
            if raw_results is None:
                error_msg = "DuckDuckGo API에서 None 응답을 반환했습니다."
                logger.error(error_msg)
                raise DuckDuckGoResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(raw_results, query)
            
            logger.info(f"DuckDuckGo 검색 완료: {len(results)}개의 결과를 찾았습니다.")
            return results
            
        except DuckDuckGoSearchException:
            # 이미 적절한 예외가 발생했으므로 그대로 전파
            raise
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"DuckDuckGo 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg, original_error=e)
            
    def _parse_search_results(self, raw_results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """
        DuckDuckGo API 응답을 SearchResult 객체 목록으로 파싱
        
        Args:
            raw_results (List[Dict[str, Any]]): DuckDuckGo API 응답
            query (str): 검색 쿼리
            
        Returns:
            List[SearchResult]: 파싱된 검색 결과 목록
        """
        results = []
        
        # 결과가 없는 경우 로그 기록
        if not raw_results:
            logger.warning(f"검색 쿼리 '{query}'에 대한 결과가 없습니다.")
            return []
            
        for index, raw_result in enumerate(raw_results):
            try:
                # 필수 필드 확인
                title = raw_result.get("title", "")
                url = raw_result.get("href", "")
                content = raw_result.get("body", "")
                
                # 제목이 없는 경우 URL에서 제목 추출 시도
                if not title:
                    try:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        path = parsed_url.path
                        if path and path != "/":
                            # 경로의 마지막 부분을 제목으로 사용
                            path_parts = path.strip("/").split("/")
                            if path_parts:
                                # 언더스코어와 하이픈을 공백으로 변환
                                title = path_parts[-1].replace("_", " ").replace("-", " ").capitalize()
                        
                        # 제목을 추출할 수 없는 경우 도메인 사용
                        if not title:
                            title = parsed_url.netloc
                    except:
                        title = "제목 없음"
                
                # 여전히 제목이 없는 경우 기본값 사용
                if not title:
                    title = "제목 없음"
                
                # URL이 없는 경우 건너뛰기
                if not url:
                    logger.warning(f"URL이 없는 검색 결과를 건너뜁니다: {title}")
                    continue
                    
                # 내용이 없는 경우 제목을 내용으로 사용
                if not content:
                    logger.warning(f"내용이 없는 검색 결과: {title}. 제목을 내용으로 사용합니다.")
                    content = title
                
                # 점수 계산 (DuckDuckGo는 기본 점수를 제공하지 않으므로 순위 기반 점수 계산)
                # 첫 번째 결과가 가장 높은 점수를 가짐
                score = 1.0 - (index * 0.05)  # 각 결과마다 0.05씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
                
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
                
                # 발행 날짜 추출 시도 (DuckDuckGo는 기본적으로 제공하지 않음)
                published_date = None
                
                # 소스 유형 추정
                source_type = self._determine_source_type(url)
                
                # 스니펫 생성 (내용의 일부를 사용)
                snippet = content
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                
                # 메타데이터 설정
                metadata = {
                    "search_engine": "duckduckgo",
                    "domain": domain,
                    "rank": index + 1,  # 검색 결과 순위
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "source_type": source_type,
                    "raw_data": raw_result
                }
                
                # 추가 메타데이터 포함 (있는 경우)
                for key in ["about", "category", "language"]:
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
        
    def _determine_source_type(self, url: str) -> str:
        """
        URL을 기반으로 소스 유형 결정
        
        Args:
            url (str): 검색 결과 URL
            
        Returns:
            str: 소스 유형 (webpage, pdf, document, video, image, code 등)
        """
        url_lower = url.lower()
        
        # 파일 확장자 기반 유형 결정
        if url_lower.endswith(".pdf"):
            return "pdf"
        elif url_lower.endswith((".doc", ".docx")):
            return "document"
        elif url_lower.endswith((".ppt", ".pptx")):
            return "presentation"
        elif url_lower.endswith((".xls", ".xlsx")):
            return "spreadsheet"
        elif url_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
            return "image"
        elif url_lower.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm")):
            return "video"
        elif url_lower.endswith((".mp3", ".wav", ".ogg", ".flac", ".aac")):
            return "audio"
        elif url_lower.endswith((".zip", ".rar", ".tar", ".gz", ".7z")):
            return "archive"
        
        # 도메인 기반 유형 결정
        if "youtube.com" in url_lower or "youtu.be" in url_lower or "vimeo.com" in url_lower:
            return "video"
        elif "github.com" in url_lower or "gitlab.com" in url_lower or "bitbucket.org" in url_lower:
            return "code"
        elif "wikipedia.org" in url_lower:
            return "encyclopedia"
        elif "scholar.google.com" in url_lower or ".edu" in url_lower:
            return "academic"
        elif "news." in url_lower or ".news" in url_lower or "cnn.com" in url_lower or "bbc." in url_lower:
            return "news"
        elif "blog." in url_lower or ".blog" in url_lower or "wordpress.com" in url_lower or "medium.com" in url_lower:
            return "blog"
        elif "amazon." in url_lower or "ebay." in url_lower or "shop" in url_lower or "store" in url_lower:
            return "shopping"
        elif "stackoverflow.com" in url_lower or "stackexchange.com" in url_lower:
            return "qa"
        
        # 기본 유형
        return "webpage"
        
    def search_with_limit(self, query: str, max_results: int = 5, region: str = "wt-wt", 
                       safesearch: str = "moderate", time_limit: Optional[int] = None) -> List[SearchResult]:
        """
        시간 제한과 결과 수 제한을 적용하여 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            time_limit (Optional[int], optional): 검색 시간 제한 (초). 기본값은 None (제한 없음).
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            DuckDuckGoConnectionError: API 서버에 연결할 수 없는 경우
            DuckDuckGoResponseError: API 서버에서 오류 응답을 반환한 경우
            DuckDuckGoTimeoutError: 검색 시간이 제한을 초과한 경우
            DuckDuckGoSearchException: 기타 검색 관련 오류
        """
        # 클라이언트 사용 가능 여부 확인
        self.ensure_client_available()
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 세이프서치 설정 검증
        if safesearch not in ["on", "moderate", "off"]:
            logger.warning(f"유효하지 않은 세이프서치 설정: {safesearch}. 'moderate'로 설정합니다.")
            safesearch = "moderate"
            
        # 시간 제한 설정
        start_time = time.time()
        
        try:
            logger.info(f"DuckDuckGo 제한 검색 수행: '{query}' (최대 결과 수: {max_results}, 시간 제한: {time_limit}초)")
            
            # DuckDuckGo API를 사용하여 검색 수행
            # 시간 제한이 있는 경우 제너레이터를 사용하여 결과를 하나씩 가져오면서 시간 체크
            raw_results = []
            
            if time_limit is not None:
                # 시간 제한이 있는 경우 제너레이터를 사용하여 결과를 하나씩 가져오면서 시간 체크
                results_generator = self.client.text(
                    query, 
                    region=region, 
                    safesearch=safesearch, 
                    max_results=max_results
                )
                
                for result in results_generator:
                    raw_results.append(result)
                    
                    # 결과 수 제한 확인
                    if len(raw_results) >= max_results:
                        logger.info(f"최대 결과 수 {max_results}에 도달했습니다.")
                        break
                        
                    # 시간 제한 확인
                    elapsed_time = time.time() - start_time
                    if elapsed_time > time_limit:
                        logger.warning(f"검색 시간 제한 {time_limit}초를 초과했습니다. 현재까지 {len(raw_results)}개의 결과를 반환합니다.")
                        break
            else:
                # 시간 제한이 없는 경우 일반 검색 수행
                raw_results = list(self.client.text(
                    query, 
                    region=region, 
                    safesearch=safesearch, 
                    max_results=max_results
                ))
            
            # 응답 검증
            if raw_results is None:
                error_msg = "DuckDuckGo API에서 None 응답을 반환했습니다."
                logger.error(error_msg)
                raise DuckDuckGoResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(raw_results, query)
            
            # 검색 시간 기록
            elapsed_time = time.time() - start_time
            logger.info(f"DuckDuckGo 제한 검색 완료: {len(results)}개의 결과를 찾았습니다. (소요 시간: {elapsed_time:.2f}초)")
            
            return results
            
        except DuckDuckGoSearchException:
            # 이미 적절한 예외가 발생했으므로 그대로 전파
            raise
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"DuckDuckGo 제한 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg, original_error=e)
            
    def get_limited_results(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """
        검색 결과 수를 제한
        
        Args:
            results (List[SearchResult]): 원본 검색 결과 목록
            max_results (int): 반환할 최대 결과 수
            
        Returns:
            List[SearchResult]: 제한된 검색 결과 목록
        """
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 전체 결과를 반환합니다.")
            return results
            
        if len(results) > max_results:
            logger.info(f"검색 결과를 {max_results}개로 제한합니다 (총 {len(results)}개).")
            return results[:max_results]
            
        return results
        
    def search_with_context(self, query: str, context: str, max_results: int = 5, region: str = "wt-wt", 
                           safesearch: str = "moderate", context_weight: float = 0.5) -> List[SearchResult]:
        """
        컨텍스트를 포함한 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            context (str): 검색 컨텍스트 (이전 대화 또는 관련 정보)
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            context_weight (float, optional): 컨텍스트 가중치 (0.0 ~ 1.0). 기본값은 0.5.
                0.0: 컨텍스트 무시, 1.0: 컨텍스트 최대 중요도
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            DuckDuckGoConnectionError: API 서버에 연결할 수 없는 경우
            DuckDuckGoResponseError: API 서버에서 오류 응답을 반환한 경우
            DuckDuckGoSearchException: 기타 검색 관련 오류
        """
        # 클라이언트 사용 가능 여부 확인
        self.ensure_client_available()
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 세이프서치 설정 검증
        if safesearch not in ["on", "moderate", "off"]:
            logger.warning(f"유효하지 않은 세이프서치 설정: {safesearch}. 'moderate'로 설정합니다.")
            safesearch = "moderate"
            
        # 컨텍스트 가중치 검증
        if context_weight < 0.0 or context_weight > 1.0:
            logger.warning(f"유효하지 않은 컨텍스트 가중치: {context_weight}. 0.5로 설정합니다.")
            context_weight = 0.5
            
        try:
            # 컨텍스트와 쿼리를 결합하여 향상된 검색 쿼리 생성 (가중치 적용)
            enhanced_query = self._enhance_query_with_context(query, context, context_weight)
            
            logger.info(f"DuckDuckGo 컨텍스트 기반 검색 수행: '{enhanced_query}' (최대 결과 수: {max_results}, 컨텍스트 가중치: {context_weight})")
            
            # DuckDuckGo API를 사용하여 검색 수행
            raw_results = list(self.client.text(
                enhanced_query, 
                region=region, 
                safesearch=safesearch, 
                max_results=max_results
            ))
            
            # 응답 검증
            if raw_results is None:
                error_msg = "DuckDuckGo API에서 None 응답을 반환했습니다."
                logger.error(error_msg)
                raise DuckDuckGoResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(raw_results, enhanced_query)
            
            # 메타데이터에 원본 쿼리와 컨텍스트 정보 추가
            for result in results:
                result.metadata["original_query"] = query
                result.metadata["context_weight"] = context_weight
                result.metadata["enhanced_query"] = enhanced_query
            
            logger.info(f"DuckDuckGo 컨텍스트 기반 검색 완료: {len(results)}개의 결과를 찾았습니다.")
            return results
            
        except DuckDuckGoSearchException:
            # 이미 적절한 예외가 발생했으므로 그대로 전파
            raise
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"DuckDuckGo 컨텍스트 기반 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg, original_error=e)
            
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
            
        # 컨텍스트가 너무 긴 경우 잘라내기 (API 제한 고려)
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
        
    def search_with_weighted_context(self, query: str, contexts: List[Dict[str, Any]], max_results: int = 5, 
                                region: str = "wt-wt", safesearch: str = "moderate") -> List[SearchResult]:
        """
        여러 컨텍스트와 가중치를 적용한 웹 검색을 수행
        
        Args:
            query (str): 검색 쿼리
            contexts (List[Dict[str, Any]]): 컨텍스트 목록. 각 항목은 다음 키를 포함해야 함:
                - 'text': 컨텍스트 텍스트
                - 'weight': 컨텍스트 가중치 (0.0 ~ 1.0)
                - 'type': 컨텍스트 유형 (선택 사항, 예: 'conversation', 'document', 'user_profile')
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            
        Returns:
            List[SearchResult]: 검색 결과 목록
            
        Raises:
            DuckDuckGoConnectionError: API 서버에 연결할 수 없는 경우
            DuckDuckGoResponseError: API 서버에서 오류 응답을 반환한 경우
            DuckDuckGoSearchException: 기타 검색 관련 오류
        """
        # 클라이언트 사용 가능 여부 확인
        self.ensure_client_available()
            
        # 최대 결과 수 검증
        if max_results < 1:
            logger.warning(f"유효하지 않은 최대 결과 수: {max_results}. 5로 설정합니다.")
            max_results = 5
            
        # 세이프서치 설정 검증
        if safesearch not in ["on", "moderate", "off"]:
            logger.warning(f"유효하지 않은 세이프서치 설정: {safesearch}. 'moderate'로 설정합니다.")
            safesearch = "moderate"
            
        # 컨텍스트 검증
        if not contexts:
            logger.warning("컨텍스트가 제공되지 않았습니다. 기본 검색을 수행합니다.")
            return self.search(query, max_results, region, safesearch)
            
        try:
            # 여러 컨텍스트를 처리하여 향상된 검색 쿼리 생성
            enhanced_query = self._enhance_query_with_multiple_contexts(query, contexts)
            
            logger.info(f"DuckDuckGo 가중치 컨텍스트 기반 검색 수행: '{enhanced_query}' (최대 결과 수: {max_results})")
            
            # DuckDuckGo API를 사용하여 검색 수행
            raw_results = list(self.client.text(
                enhanced_query, 
                region=region, 
                safesearch=safesearch, 
                max_results=max_results
            ))
            
            # 응답 검증
            if raw_results is None:
                error_msg = "DuckDuckGo API에서 None 응답을 반환했습니다."
                logger.error(error_msg)
                raise DuckDuckGoResponseError(error_msg)
                
            # 결과 파싱 및 변환
            results = self._parse_search_results(raw_results, enhanced_query)
            
            # 메타데이터에 원본 쿼리와 컨텍스트 정보 추가
            for result in results:
                result.metadata["original_query"] = query
                result.metadata["enhanced_query"] = enhanced_query
                result.metadata["contexts_count"] = len(contexts)
            
            logger.info(f"DuckDuckGo 가중치 컨텍스트 기반 검색 완료: {len(results)}개의 결과를 찾았습니다.")
            return results
            
        except DuckDuckGoSearchException:
            # 이미 적절한 예외가 발생했으므로 그대로 전파
            raise
            
        except Exception as e:
            # 기타 예상치 못한 오류
            error_msg = f"DuckDuckGo 가중치 컨텍스트 기반 검색 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg, original_error=e)
            
    def _enhance_query_with_multiple_contexts(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        여러 컨텍스트를 사용하여 검색 쿼리를 향상
        
        Args:
            query (str): 원본 검색 쿼리
            contexts (List[Dict[str, Any]]): 컨텍스트 목록
            
        Returns:
            str: 향상된 검색 쿼리
        """
        if not contexts:
            return query
            
        # 각 컨텍스트에서 키워드 추출 및 가중치 적용
        weighted_keywords = []
        
        for context_item in contexts:
            # 필수 필드 확인
            if 'text' not in context_item or not context_item['text']:
                continue
                
            context_text = context_item['text']
            context_weight = context_item.get('weight', 0.5)
            context_type = context_item.get('type', 'general')
            
            # 가중치 검증
            if context_weight <= 0.0:
                continue
                
            if context_weight > 1.0:
                context_weight = 1.0
                
            # 컨텍스트 유형에 따른 가중치 조정
            if context_type == 'conversation':
                # 대화 컨텍스트는 더 높은 가중치를 가질 수 있음
                context_weight = min(1.0, context_weight * 1.2)
            elif context_type == 'user_profile':
                # 사용자 프로필 컨텍스트는 더 낮은 가중치를 가질 수 있음
                context_weight = context_weight * 0.8
                
            # 컨텍스트에서 키워드 추출
            keywords = self._extract_keywords_from_context(context_text)
            
            if keywords:
                # 키워드 목록 생성
                keyword_list = keywords.split()
                
                # 가중치에 따라 키워드 수 결정
                num_keywords = max(1, int(len(keyword_list) * context_weight))
                selected_keywords = keyword_list[:num_keywords]
                
                # 가중치가 높을수록 더 많은 키워드 포함
                weighted_keywords.extend(selected_keywords)
                
        # 중복 키워드 제거 및 상위 키워드 선택
        unique_keywords = list(set(weighted_keywords))
        
        # 키워드가 너무 많은 경우 제한
        max_keywords = 10
        if len(unique_keywords) > max_keywords:
            unique_keywords = unique_keywords[:max_keywords]
            
        # 키워드가 없는 경우 원본 쿼리 반환
        if not unique_keywords:
            return query
            
        # 키워드를 쿼리에 추가
        enhanced_query = f"{query} {' '.join(unique_keywords)}"
        logger.info(f"여러 컨텍스트로 쿼리가 향상되었습니다: '{query}' -> '{enhanced_query}'")
        return enhanced_query
        
    def adjust_context_weight(self, base_weight: float, context_type: str, context_length: int, 
                             context_age: Optional[int] = None) -> float:
        """
        컨텍스트 유형, 길이, 나이에 따라 가중치 조정
        
        Args:
            base_weight (float): 기본 가중치 (0.0 ~ 1.0)
            context_type (str): 컨텍스트 유형 ('conversation', 'document', 'user_profile' 등)
            context_length (int): 컨텍스트 길이 (문자 수)
            context_age (Optional[int], optional): 컨텍스트 나이 (초). 기본값은 None.
            
        Returns:
            float: 조정된 가중치 (0.0 ~ 1.0)
        """
        # 기본 가중치 검증
        weight = max(0.0, min(1.0, base_weight))
        
        # 컨텍스트 유형에 따른 조정
        type_multipliers = {
            'conversation': 1.2,  # 대화 컨텍스트는 더 중요
            'document': 1.0,      # 문서 컨텍스트는 기본 가중치
            'user_profile': 0.8,  # 사용자 프로필은 덜 중요
            'search_history': 1.1 # 검색 기록은 약간 더 중요
        }
        
        # 컨텍스트 유형에 따른 가중치 조정
        type_multiplier = type_multipliers.get(context_type, 1.0)
        weight *= type_multiplier
        
        # 컨텍스트 길이에 따른 조정
        # 너무 짧은 컨텍스트는 덜 중요, 적당한 길이의 컨텍스트는 더 중요
        if context_length < 20:
            # 매우 짧은 컨텍스트
            weight *= 0.7
        elif context_length < 50:
            # 짧은 컨텍스트
            weight *= 0.9
        elif context_length > 500:
            # 매우 긴 컨텍스트
            weight *= 0.8
        elif context_length > 200:
            # 긴 컨텍스트
            weight *= 0.95
        else:
            # 적당한 길이의 컨텍스트
            weight *= 1.1
            
        # 컨텍스트 나이에 따른 조정 (제공된 경우)
        if context_age is not None:
            # 오래된 컨텍스트는 덜 중요
            if context_age > 86400:  # 1일 이상
                weight *= 0.7
            elif context_age > 3600:  # 1시간 이상
                weight *= 0.9
            elif context_age < 60:    # 1분 미만
                weight *= 1.2
                
        # 최종 가중치 범위 제한
        weight = max(0.1, min(1.0, weight))
        
        return weight   
 def parse_news_results(self, raw_results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """
        DuckDuckGo 뉴스 검색 결과를 SearchResult 객체 목록으로 파싱
        
        Args:
            raw_results (List[Dict[str, Any]]): DuckDuckGo 뉴스 API 응답
            query (str): 검색 쿼리
            
        Returns:
            List[SearchResult]: 파싱된 검색 결과 목록
        """
        results = []
        
        # 결과가 없는 경우 로그 기록
        if not raw_results:
            logger.warning(f"뉴스 검색 쿼리 '{query}'에 대한 결과가 없습니다.")
            return []
            
        for index, raw_result in enumerate(raw_results):
            try:
                # 필수 필드 확인
                title = raw_result.get("title", "제목 없음")
                url = raw_result.get("url", "")
                content = raw_result.get("body", "")
                
                # URL이 없는 경우 href 필드 확인
                if not url:
                    url = raw_result.get("href", "")
                
                # 여전히 URL이 없는 경우 건너뛰기
                if not url:
                    logger.warning(f"URL이 없는 뉴스 검색 결과를 건너뜁니다: {title}")
                    continue
                    
                # 내용이 없는 경우 snippet 필드 확인
                if not content:
                    content = raw_result.get("snippet", "")
                
                # 여전히 내용이 없는 경우 제목을 내용으로 사용
                if not content:
                    logger.warning(f"내용이 없는 뉴스 검색 결과: {title}. 제목을 내용으로 사용합니다.")
                    content = title
                
                # 점수 계산 (뉴스 결과는 시간 기반으로 더 높은 점수 부여)
                score = 1.0 - (index * 0.03)  # 각 결과마다 0.03씩 감소
                score = max(0.6, min(1.0, score))  # 0.6에서 1.0 사이로 제한
                
                # 발행 날짜 추출 시도
                published_date = None
                if "date" in raw_result:
                    try:
                        # 다양한 날짜 형식 파싱 시도
                        from dateutil import parser
                        published_date = parser.parse(str(raw_result["date"]))
                    except (ImportError, ValueError, TypeError):
                        logger.warning(f"발행 날짜를 파싱할 수 없습니다: {raw_result.get('date')}")
                
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
                
                # 메타데이터 설정
                metadata = {
                    "search_engine": "duckduckgo",
                    "search_type": "news",
                    "domain": domain,
                    "rank": index + 1,  # 검색 결과 순위
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "raw_data": raw_result
                }
                
                # 추가 메타데이터 포함 (있는 경우)
                for key in ["source", "category", "language", "image"]:
                    if key in raw_result:
                        metadata[key] = raw_result[key]
                
                # SearchResult 객체 생성
                result = SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=score,
                    published_date=published_date,
                    snippet=content[:200] + "..." if len(content) > 200 else content,
                    source_type="news",
                    metadata=metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"뉴스 검색 결과 파싱 중 오류 발생: {str(e)}")
                continue
        
        # 결과 정렬 (점수 기준 내림차순)
        results.sort(key=lambda x: x.score, reverse=True)
                
        return results
        
    def parse_images_results(self, raw_results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """
        DuckDuckGo 이미지 검색 결과를 SearchResult 객체 목록으로 파싱
        
        Args:
            raw_results (List[Dict[str, Any]]): DuckDuckGo 이미지 API 응답
            query (str): 검색 쿼리
            
        Returns:
            List[SearchResult]: 파싱된 검색 결과 목록
        """
        results = []
        
        # 결과가 없는 경우 로그 기록
        if not raw_results:
            logger.warning(f"이미지 검색 쿼리 '{query}'에 대한 결과가 없습니다.")
            return []
            
        for index, raw_result in enumerate(raw_results):
            try:
                # 필수 필드 확인
                title = raw_result.get("title", "이미지 제목 없음")
                image_url = raw_result.get("image", "")
                source_url = raw_result.get("url", "")
                
                # 이미지 URL이 없는 경우 건너뛰기
                if not image_url:
                    logger.warning(f"이미지 URL이 없는 검색 결과를 건너뜁니다: {title}")
                    continue
                    
                # 소스 URL이 없는 경우 이미지 URL을 소스 URL로 사용
                if not source_url:
                    source_url = image_url
                
                # 내용 생성 (이미지 설명)
                content = f"이미지: {title}"
                if "height" in raw_result and "width" in raw_result:
                    content += f" (크기: {raw_result['width']}x{raw_result['height']})"
                
                # 점수 계산
                score = 1.0 - (index * 0.04)  # 각 결과마다 0.04씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
                
                # 도메인 추출
                domain = ""
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(source_url)
                    domain = parsed_url.netloc
                except (ImportError, ValueError, AttributeError):
                    # URL에서 도메인 추출 실패 시 간단한 방법 사용
                    domain_parts = source_url.split("//")
                    if len(domain_parts) > 1:
                        domain = domain_parts[1].split("/")[0]
                
                # 메타데이터 설정
                metadata = {
                    "search_engine": "duckduckgo",
                    "search_type": "images",
                    "domain": domain,
                    "rank": index + 1,  # 검색 결과 순위
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "image_url": image_url,
                    "raw_data": raw_result
                }
                
                # 추가 메타데이터 포함 (있는 경우)
                for key in ["height", "width", "thumbnail", "source"]:
                    if key in raw_result:
                        metadata[key] = raw_result[key]
                
                # SearchResult 객체 생성
                result = SearchResult(
                    title=title,
                    url=source_url,
                    content=content,
                    score=score,
                    snippet=title,
                    source_type="image",
                    metadata=metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"이미지 검색 결과 파싱 중 오류 발생: {str(e)}")
                continue
        
        # 결과 정렬 (점수 기준 내림차순)
        results.sort(key=lambda x: x.score, reverse=True)
                
        return results
        
    def parse_videos_results(self, raw_results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """
        DuckDuckGo 비디오 검색 결과를 SearchResult 객체 목록으로 파싱
        
        Args:
            raw_results (List[Dict[str, Any]]): DuckDuckGo 비디오 API 응답
            query (str): 검색 쿼리
            
        Returns:
            List[SearchResult]: 파싱된 검색 결과 목록
        """
        results = []
        
        # 결과가 없는 경우 로그 기록
        if not raw_results:
            logger.warning(f"비디오 검색 쿼리 '{query}'에 대한 결과가 없습니다.")
            return []
            
        for index, raw_result in enumerate(raw_results):
            try:
                # 필수 필드 확인
                title = raw_result.get("title", "비디오 제목 없음")
                url = raw_result.get("url", "")
                content = raw_result.get("description", "")
                
                # URL이 없는 경우 건너뛰기
                if not url:
                    logger.warning(f"URL이 없는 비디오 검색 결과를 건너뜁니다: {title}")
                    continue
                    
                # 내용이 없는 경우 제목을 내용으로 사용
                if not content:
                    content = title
                
                # 점수 계산
                score = 1.0 - (index * 0.04)  # 각 결과마다 0.04씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
                
                # 발행 날짜 추출 시도
                published_date = None
                if "published" in raw_result:
                    try:
                        # 다양한 날짜 형식 파싱 시도
                        from dateutil import parser
                        published_date = parser.parse(str(raw_result["published"]))
                    except (ImportError, ValueError, TypeError):
                        logger.warning(f"발행 날짜를 파싱할 수 없습니다: {raw_result.get('published')}")
                
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
                
                # 메타데이터 설정
                metadata = {
                    "search_engine": "duckduckgo",
                    "search_type": "videos",
                    "domain": domain,
                    "rank": index + 1,  # 검색 결과 순위
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "raw_data": raw_result
                }
                
                # 추가 메타데이터 포함 (있는 경우)
                for key in ["duration", "publisher", "thumbnail", "viewCount"]:
                    if key in raw_result:
                        metadata[key] = raw_result[key]
                
                # SearchResult 객체 생성
                result = SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=score,
                    published_date=published_date,
                    snippet=content[:200] + "..." if len(content) > 200 else content,
                    source_type="video",
                    metadata=metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"비디오 검색 결과 파싱 중 오류 발생: {str(e)}")
                continue
        
        # 결과 정렬 (점수 기준 내림차순)
        results.sort(key=lambda x: x.score, reverse=True)
                
        return results    @static
method
    def convert_to_search_result(raw_result: Dict[str, Any], query: str = "", rank: int = 0) -> Optional[SearchResult]:
        """
        단일 DuckDuckGo API 결과를 SearchResult 객체로 변환
        
        Args:
            raw_result (Dict[str, Any]): DuckDuckGo API 결과 항목
            query (str, optional): 검색 쿼리. 기본값은 빈 문자열.
            rank (int, optional): 검색 결과 순위. 기본값은 0.
            
        Returns:
            Optional[SearchResult]: 변환된 SearchResult 객체 또는 변환 실패 시 None
        """
        try:
            # 필수 필드 확인
            title = raw_result.get("title", "")
            url = raw_result.get("href", "")
            
            # URL이 없는 경우 다른 필드 확인
            if not url:
                url = raw_result.get("url", "")
            
            # 여전히 URL이 없는 경우 None 반환
            if not url:
                logger.warning(f"URL이 없는 검색 결과를 변환할 수 없습니다: {title}")
                return None
                
            # 제목이 없는 경우 URL에서 제목 추출 시도
            if not title:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    path = parsed_url.path
                    if path and path != "/":
                        # 경로의 마지막 부분을 제목으로 사용
                        path_parts = path.strip("/").split("/")
                        if path_parts:
                            # 언더스코어와 하이픈을 공백으로 변환
                            title = path_parts[-1].replace("_", " ").replace("-", " ").capitalize()
                    
                    # 제목을 추출할 수 없는 경우 도메인 사용
                    if not title:
                        title = parsed_url.netloc
                except:
                    title = "제목 없음"
            
            # 여전히 제목이 없는 경우 기본값 사용
            if not title:
                title = "제목 없음"
                
            # 내용 확인
            content = raw_result.get("body", "")
            
            # 내용이 없는 경우 다른 필드 확인
            if not content:
                content = raw_result.get("snippet", "")
                
            # 여전히 내용이 없는 경우 제목을 내용으로 사용
            if not content:
                content = title
            
            # 점수 계산
            score = 1.0
            if rank > 0:
                score = 1.0 - (rank * 0.05)  # 각 결과마다 0.05씩 감소
                score = max(0.5, min(1.0, score))  # 0.5에서 1.0 사이로 제한
            
            # 발행 날짜 추출 시도
            published_date = None
            for date_field in ["date", "published", "publishedDate", "datePublished"]:
                if date_field in raw_result:
                    try:
                        # 다양한 날짜 형식 파싱 시도
                        from dateutil import parser
                        published_date = parser.parse(str(raw_result[date_field]))
                        break
                    except (ImportError, ValueError, TypeError):
                        continue
            
            # 소스 유형 결정
            source_type = raw_result.get("source_type", "")
            if not source_type:
                # URL 기반으로 소스 유형 추정
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
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
            
            # 스니펫 생성
            snippet = content
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            
            # 메타데이터 설정
            metadata = {
                "search_engine": "duckduckgo",
                "domain": domain,
                "rank": rank,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "raw_data": raw_result
            }
            
            # 추가 메타데이터 포함 (있는 경우)
            for key in ["category", "language", "source", "about"]:
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
        DuckDuckGo API 결과 목록을 SearchResult 객체 목록으로 변환
        
        Args:
            raw_results (List[Dict[str, Any]]): DuckDuckGo API 결과 목록
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
        
    def merge_search_results(self, results_list: List[List[SearchResult]], max_results: int = 10) -> List[SearchResult]:
        """
        여러 검색 결과 목록을 병합
        
        Args:
            results_list (List[List[SearchResult]]): 검색 결과 목록의 목록
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 10.
            
        Returns:
            List[SearchResult]: 병합된 검색 결과 목록
        """
        # 모든 결과를 하나의 목록으로 병합
        all_results = []
        for results in results_list:
            all_results.extend(results)
            
        # 중복 URL 제거 (첫 번째 발견된 결과 유지)
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            url = str(result.url)
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
                
        # 점수 기준으로 정렬
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # 최대 결과 수 제한
        if len(unique_results) > max_results:
            unique_results = unique_results[:max_results]
            
        return unique_results    
def _handle_api_error(self, error: Exception, query: str) -> None:
        """
        DuckDuckGo API 오류 처리 및 적절한 예외 발생
        
        Args:
            error (Exception): 원본 예외
            query (str): 검색 쿼리
            
        Raises:
            DuckDuckGoRateLimitError: 속도 제한 오류 발생 시
            DuckDuckGoTimeoutError: 타임아웃 오류 발생 시
            DuckDuckGoConnectionError: 연결 오류 발생 시
            DuckDuckGoResponseError: 응답 오류 발생 시
            DuckDuckGoSearchException: 기타 오류 발생 시
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # 속도 제한 오류 처리
        if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
            # 재시도 시간 추출 시도
            retry_after = None
            if hasattr(error, "headers") and "retry-after" in error.headers:
                try:
                    retry_after = int(error.headers["retry-after"])
                except (ValueError, TypeError):
                    pass
                    
            error_msg = f"DuckDuckGo API 속도 제한 오류: {str(error)}"
            logger.error(error_msg)
            raise DuckDuckGoRateLimitError(error_msg, retry_after=retry_after, original_error=error)
            
        # 타임아웃 오류 처리
        elif "timeout" in error_str or "timed out" in error_str:
            error_msg = f"DuckDuckGo API 타임아웃 오류: {str(error)}"
            logger.error(error_msg)
            raise DuckDuckGoTimeoutError(error_msg, original_error=error)
            
        # 연결 오류 처리
        elif "connection" in error_str or "connect" in error_str or error_type in ["ConnectionError", "ConnectionRefusedError"]:
            error_msg = f"DuckDuckGo API 연결 오류: {str(error)}"
            logger.error(error_msg)
            raise DuckDuckGoConnectionError(error_msg, original_error=error)
            
        # 응답 오류 처리
        elif "response" in error_str or "status" in error_str or error_type in ["HTTPError", "ResponseError"]:
            # 상태 코드 추출 시도
            status_code = None
            response_body = None
            
            if hasattr(error, "status_code"):
                status_code = error.status_code
            elif hasattr(error, "code"):
                status_code = error.code
                
            if hasattr(error, "response") and hasattr(error.response, "text"):
                response_body = error.response.text
                
            error_msg = f"DuckDuckGo API 응답 오류: {str(error)}"
            logger.error(error_msg)
            raise DuckDuckGoResponseError(error_msg, status_code=status_code, response_body=response_body, original_error=error)
            
        # 기타 오류 처리
        else:
            error_msg = f"DuckDuckGo 검색 중 예상치 못한 오류가 발생했습니다: {str(error)}"
            logger.error(error_msg)
            raise DuckDuckGoSearchException(error_msg, original_error=error)
            
    def handle_search_error(self, error: Exception, query: str) -> List[SearchResult]:
        """
        검색 오류를 처리하고 빈 결과 목록 반환
        
        Args:
            error (Exception): 발생한 예외
            query (str): 검색 쿼리
            
        Returns:
            List[SearchResult]: 빈 검색 결과 목록
        """
        try:
            # 오류 유형에 따른 로깅
            if isinstance(error, DuckDuckGoRateLimitError):
                retry_after = error.retry_after if hasattr(error, "retry_after") else "알 수 없음"
                logger.error(f"DuckDuckGo 속도 제한 오류 (재시도 시간: {retry_after}초): {str(error)}")
            elif isinstance(error, DuckDuckGoTimeoutError):
                logger.error(f"DuckDuckGo 타임아웃 오류: {str(error)}")
            elif isinstance(error, DuckDuckGoConnectionError):
                logger.error(f"DuckDuckGo 연결 오류: {str(error)}")
            elif isinstance(error, DuckDuckGoResponseError):
                status_code = error.status_code if hasattr(error, "status_code") else "알 수 없음"
                logger.error(f"DuckDuckGo 응답 오류 (상태 코드: {status_code}): {str(error)}")
            else:
                logger.error(f"DuckDuckGo 검색 중 예상치 못한 오류: {str(error)}")
                
            # 오류 정보를 포함한 빈 결과 목록 반환
            return []
            
        except Exception as e:
            # 오류 처리 중 발생한 예외 처리
            logger.error(f"오류 처리 중 추가 예외 발생: {str(e)}")
            return []
            
    def search_safely(self, query: str, max_results: int = 5, region: str = "wt-wt", 
                     safesearch: str = "moderate") -> List[SearchResult]:
        """
        예외 처리를 포함한 안전한 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            
        Returns:
            List[SearchResult]: 검색 결과 목록 또는 오류 발생 시 빈 목록
        """
        try:
            return self.search(query, max_results, region, safesearch)
        except Exception as e:
            return self.handle_search_error(e, query)
            
    def search_with_context_safely(self, query: str, context: str, max_results: int = 5, 
                                  region: str = "wt-wt", safesearch: str = "moderate", 
                                  context_weight: float = 0.5) -> List[SearchResult]:
        """
        예외 처리를 포함한 안전한 컨텍스트 기반 검색 수행
        
        Args:
            query (str): 검색 쿼리
            context (str): 검색 컨텍스트
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            context_weight (float, optional): 컨텍스트 가중치 (0.0 ~ 1.0). 기본값은 0.5.
            
        Returns:
            List[SearchResult]: 검색 결과 목록 또는 오류 발생 시 빈 목록
        """
        try:
            return self.search_with_context(query, context, max_results, region, safesearch, context_weight)
        except Exception as e:
            return self.handle_search_error(e, query)    def s
earch_with_fallback(self, query: str, max_results: int = 5, region: str = "wt-wt", 
                          safesearch: str = "moderate", fallback_message: str = None) -> List[SearchResult]:
        """
        실패 시 대체 결과를 제공하는 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            fallback_message (str, optional): 실패 시 표시할 메시지. 기본값은 None.
            
        Returns:
            List[SearchResult]: 검색 결과 목록 또는 실패 시 대체 결과
        """
        try:
            # 일반 검색 시도
            results = self.search(query, max_results, region, safesearch)
            
            # 결과가 있으면 반환
            if results:
                return results
                
            # 결과가 없으면 대체 결과 생성
            logger.warning(f"검색 쿼리 '{query}'에 대한 결과가 없습니다. 대체 결과를 생성합니다.")
            return self._create_fallback_result(query, fallback_message)
            
        except Exception as e:
            # 오류 발생 시 대체 결과 생성
            logger.error(f"검색 중 오류 발생: {str(e)}. 대체 결과를 생성합니다.")
            return self._create_fallback_result(query, fallback_message, error=str(e))
            
    def search_with_retry(self, query: str, max_results: int = 5, region: str = "wt-wt", 
                         safesearch: str = "moderate", max_retries: int = 3, 
                         retry_delay: float = 1.0) -> List[SearchResult]:
        """
        자동 재시도 기능이 있는 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            region (str, optional): 검색 지역 코드. 기본값은 "wt-wt" (전 세계).
            safesearch (str, optional): 세이프서치 설정 ("on", "moderate", "off"). 기본값은 "moderate".
            max_retries (int, optional): 최대 재시도 횟수. 기본값은 3.
            retry_delay (float, optional): 재시도 간 지연 시간 (초). 기본값은 1.0.
            
        Returns:
            List[SearchResult]: 검색 결과 목록 또는 모든 재시도 실패 시 빈 목록
        """
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # 검색 시도
                results = self.search(query, max_results, region, safesearch)
                
                # 결과가 있으면 반환
                if results:
                    if retries > 0:
                        logger.info(f"{retries}번의 재시도 후 검색 성공")
                    return results
                    
                # 결과가 없으면 재시도
                logger.warning(f"검색 쿼리 '{query}'에 대한 결과가 없습니다. 재시도 {retries + 1}/{max_retries}")
                
            except DuckDuckGoRateLimitError as e:
                # 속도 제한 오류는 더 긴 지연 시간 필요
                logger.warning(f"속도 제한 오류 발생: {str(e)}. 재시도 {retries + 1}/{max_retries}")
                last_error = e
                
                # 서버에서 제공한 재시도 시간이 있으면 사용
                if hasattr(e, "retry_after") and e.retry_after:
                    time.sleep(e.retry_after)
                else:
                    # 지수 백오프 적용
                    time.sleep(retry_delay * (2 ** retries))
                    
            except (DuckDuckGoConnectionError, DuckDuckGoTimeoutError) as e:
                # 연결 오류 또는 타임아웃은 재시도
                logger.warning(f"연결 오류 또는 타임아웃 발생: {str(e)}. 재시도 {retries + 1}/{max_retries}")
                last_error = e
                
                # 지수 백오프 적용
                time.sleep(retry_delay * (2 ** retries))
                
            except Exception as e:
                # 기타 오류는 재시도하지 않고 바로 실패
                logger.error(f"검색 중 복구 불가능한 오류 발생: {str(e)}")
                return self._create_fallback_result(query, error=str(e))
                
            retries += 1
            
        # 모든 재시도 실패 시
        logger.error(f"최대 재시도 횟수 {max_retries}회를 초과했습니다. 마지막 오류: {str(last_error) if last_error else '결과 없음'}")
        return self._create_fallback_result(query, error=str(last_error) if last_error else "최대 재시도 횟수 초과")
        
    def _create_fallback_result(self, query: str, fallback_message: str = None, error: str = None) -> List[SearchResult]:
        """
        실패 시 대체 결과 생성
        
        Args:
            query (str): 검색 쿼리
            fallback_message (str, optional): 실패 시 표시할 메시지. 기본값은 None.
            error (str, optional): 발생한 오류 메시지. 기본값은 None.
            
        Returns:
            List[SearchResult]: 대체 결과를 포함한 목록
        """
        # 기본 대체 메시지
        if not fallback_message:
            fallback_message = f"검색 쿼리 '{query}'에 대한 결과를 찾을 수 없습니다."
            
        # 오류 정보가 있으면 추가
        if error:
            content = f"{fallback_message}\n\n오류 정보: {error}"
        else:
            content = fallback_message
            
        # 대체 결과 생성
        fallback_result = SearchResult(
            title=f"'{query}' 검색 실패",
            url="https://duckduckgo.com/?q=" + query.replace(" ", "+"),
            content=content,
            score=0.0,
            snippet=fallback_message,
            source_type="error",
            metadata={
                "search_engine": "duckduckgo",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "is_fallback": True
            }
        )
        
        return [fallback_result]
        
    def search_with_backup_strategy(self, query: str, max_results: int = 5, 
                                   backup_strategies: List[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        여러 백업 전략을 사용한 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int, optional): 반환할 최대 결과 수. 기본값은 5.
            backup_strategies (List[Dict[str, Any]], optional): 백업 전략 목록. 기본값은 None.
                각 전략은 다음 키를 포함할 수 있음:
                - 'region': 검색 지역 코드
                - 'safesearch': 세이프서치 설정
                - 'modified_query': 수정된 쿼리
            
        Returns:
            List[SearchResult]: 검색 결과 목록 또는 모든 전략 실패 시 대체 결과
        """
        # 기본 전략 설정
        if not backup_strategies:
            backup_strategies = [
                {"region": "wt-wt", "safesearch": "moderate"},  # 기본 전략
                {"region": "us-en", "safesearch": "moderate"},  # 미국 영어 지역
                {"modified_query": f"{query} information"},     # 쿼리 수정 전략
                {"modified_query": f"about {query}"}            # 다른 쿼리 수정 전략
            ]
            
        # 첫 번째 전략 (기본 검색)
        try:
            logger.info(f"기본 전략으로 '{query}' 검색 시도")
            results = self.search(query, max_results)
            
            # 결과가 있으면 반환
            if results:
                return results
                
            logger.warning(f"기본 전략으로 결과를 찾을 수 없습니다. 백업 전략 시도")
            
        except Exception as e:
            logger.warning(f"기본 전략 실패: {str(e)}. 백업 전략 시도")
            
        # 백업 전략 시도
        for i, strategy in enumerate(backup_strategies):
            try:
                # 전략 파라미터 추출
                region = strategy.get("region", "wt-wt")
                safesearch = strategy.get("safesearch", "moderate")
                modified_query = strategy.get("modified_query", query)
                
                logger.info(f"백업 전략 {i + 1}/{len(backup_strategies)} 시도: 쿼리='{modified_query}', 지역={region}")
                
                # 백업 전략으로 검색
                results = self.search(modified_query, max_results, region, safesearch)
                
                # 결과가 있으면 반환
                if results:
                    logger.info(f"백업 전략 {i + 1}로 {len(results)}개의 결과를 찾았습니다.")
                    
                    # 메타데이터에 백업 전략 정보 추가
                    for result in results:
                        result.metadata["backup_strategy"] = i + 1
                        result.metadata["original_query"] = query
                        if modified_query != query:
                            result.metadata["modified_query"] = modified_query
                            
                    return results
                    
            except Exception as e:
                logger.warning(f"백업 전략 {i + 1} 실패: {str(e)}")
                continue
                
        # 모든 전략 실패 시 대체 결과 생성
        logger.error(f"모든 검색 전략이 실패했습니다. 대체 결과를 생성합니다.")
        return self._create_fallback_result(query, fallback_message=f"'{query}'에 대한 검색 결과를 찾을 수 없습니다. 모든 검색 전략이 실패했습니다.")