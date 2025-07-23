# src/tools/data_processor.py

"""
데이터 처리 유틸리티 구현
DataProcessor 클래스 구현
"""

import re
import logging
import html
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.models.search import SearchResult

# 로깅 설정
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    검색 결과 및 텍스트 데이터 처리 유틸리티
    
    텍스트 정제, 중복 탐지, 데이터 형식 표준화, 메타데이터 정규화,
    도메인 신뢰도 평가, 콘텐츠 품질 평가, 검색 결과 통합 등의 기능을 제공합니다.
    
    Attributes:
        nlp: NLP 도구 (선택적)
        trusted_domains (List[str]): 신뢰할 수 있는 도메인 목록
        domain_scores (Dict[str, float]): 도메인별 신뢰도 점수
    """
    
    def __init__(self, use_nlp: bool = False):
        """
        DataProcessor 초기화
        
        Args:
            use_nlp (bool, optional): NLP 도구 사용 여부. 기본값은 False.
        """
        self.nlp = None
        
        # NLP 도구 초기화 (선택적)
        if use_nlp:
            self._initialize_nlp()
            
        # 신뢰할 수 있는 도메인 목록 (예시)
        self.trusted_domains = [
            "wikipedia.org",
            "github.com",
            "python.org",
            "stackoverflow.com",
            "arxiv.org",
            "scholar.google.com",
            "edu",  # .edu 도메인
            "gov",  # .gov 도메인
        ]
        
        # 도메인별 신뢰도 점수 (0.0 ~ 1.0)
        self.domain_scores = {
            "wikipedia.org": 0.9,
            "github.com": 0.85,
            "python.org": 0.9,
            "stackoverflow.com": 0.8,
            "arxiv.org": 0.95,
            "scholar.google.com": 0.9,
        }
        
        logger.info("DataProcessor 초기화 완료")
    
    def _initialize_nlp(self) -> None:
        """
        NLP 도구 초기화
        
        필요에 따라 spaCy, NLTK 등의 NLP 라이브러리를 초기화합니다.
        """
        try:
            # spaCy 사용 시
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NLP 도구 초기화 완료")
            except ImportError:
                logger.warning("spaCy를 설치하지 않았거나 모델이 없습니다. 'pip install spacy' 및 'python -m spacy download en_core_web_sm'을 실행하세요.")
                
                # NLTK 대체 사용 시
                try:
                    import nltk
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    logger.info("NLTK NLP 도구 초기화 완료")
                    self.nlp = "nltk"  # NLTK 사용 표시
                except ImportError:
                    logger.warning("NLTK를 설치하지 않았습니다. 'pip install nltk'를 실행하세요.")
                    self.nlp = None
        except Exception as e:
            logger.error(f"NLP 도구 초기화 실패: {str(e)}")
            self.nlp = None
            
    def clean_text(self, text: str) -> str:
        """
        텍스트 정제
        
        HTML 태그 제거, 특수 문자 정리, 여러 공백 제거 등의 작업을 수행합니다.
        
        Args:
            text (str): 정제할 텍스트
            
        Returns:
            str: 정제된 텍스트
        """
        if not text:
            return ""
            
        # HTML 태그 제거
        cleaned_text = self.remove_html_tags(text)
        
        # HTML 엔티티 디코딩
        cleaned_text = html.unescape(cleaned_text)
        
        # 특수 문자 정리
        cleaned_text = self.clean_special_chars(cleaned_text)
        
        # 여러 공백 제거
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    def remove_html_tags(self, text: str) -> str:
        """
        HTML 태그 제거
        
        Args:
            text (str): HTML 태그를 포함한 텍스트
            
        Returns:
            str: HTML 태그가 제거된 텍스트
        """
        if not text:
            return ""
            
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        
        # 스크립트 및 스타일 블록 제거
        clean_text = re.sub(r'<script.*?</script>', ' ', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<style.*?</style>', ' ', clean_text, flags=re.DOTALL)
        
        # 주석 제거
        clean_text = re.sub(r'<!--.*?-->', ' ', clean_text, flags=re.DOTALL)
        
        # 여러 공백 제거
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
        
    def clean_special_chars(self, text: str) -> str:
        """
        특수 문자 정리
        
        Args:
            text (str): 특수 문자를 포함한 텍스트
            
        Returns:
            str: 특수 문자가 정리된 텍스트
        """
        if not text:
            return ""
            
        # 유니코드 제어 문자 제거
        clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # 불필요한 구두점 정리 (연속된 구두점을 하나로)
        clean_text = re.sub(r'[.]{2,}', '...', clean_text)  # 연속된 마침표는 줄임표로
        clean_text = re.sub(r'[!]{2,}', '!', clean_text)    # 연속된 느낌표는 하나로
        clean_text = re.sub(r'[?]{2,}', '?', clean_text)    # 연속된 물음표는 하나로
        
        # 특수 유니코드 공백 문자를 일반 공백으로 변환
        clean_text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', clean_text)
        
        # 여러 공백 제거
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
        
    def detect_duplicates(self, results: List[SearchResult], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        중복 검색 결과 탐지
        
        Args:
            results (List[SearchResult]): 검색 결과 목록
            threshold (float, optional): 유사도 임계값. 기본값은 0.8.
            
        Returns:
            List[Tuple[int, int, float]]: 중복 쌍 목록 (인덱스1, 인덱스2, 유사도)
        """
        if not results or len(results) < 2:
            return []
            
        duplicates = []
        
        # 모든 결과 쌍에 대해 유사도 계산
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # URL이 동일한 경우 중복으로 간주
                if results[i].url == results[j].url:
                    duplicates.append((i, j, 1.0))
                    continue
                    
                # 제목과 내용 유사도 계산
                title_similarity = self.calculate_text_similarity(
                    results[i].title, 
                    results[j].title
                )
                
                content_similarity = self.calculate_text_similarity(
                    results[i].content[:500],  # 내용이 길 수 있으므로 앞부분만 비교
                    results[j].content[:500]
                )
                
                # 제목과 내용의 유사도 평균
                avg_similarity = (title_similarity * 0.4 + content_similarity * 0.6)
                
                # 임계값 이상인 경우 중복으로 간주
                if avg_similarity >= threshold:
                    duplicates.append((i, j, avg_similarity))
                    
        return duplicates
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 유사도 계산
        
        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트
            
        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        if not text1 or not text2:
            return 0.0
            
        # 텍스트 정제
        text1 = self.clean_text(text1.lower())
        text2 = self.clean_text(text2.lower())
        
        # 텍스트가 완전히 동일한 경우
        if text1 == text2:
            return 1.0
            
        # NLP 도구가 있는 경우 더 정교한 유사도 계산
        if self.nlp and isinstance(self.nlp, object) and self.nlp != "nltk":
            try:
                # spaCy 사용
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                return doc1.similarity(doc2)
            except Exception as e:
                logger.warning(f"spaCy 유사도 계산 실패: {str(e)}")
                # 실패 시 기본 방법으로 대체
        
        # 기본 방법: 자카드 유사도 계산
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def filter_duplicates(self, results: List[SearchResult], threshold: float = 0.8) -> List[SearchResult]:
        """
        중복 검색 결과 필터링
        
        Args:
            results (List[SearchResult]): 검색 결과 목록
            threshold (float, optional): 유사도 임계값. 기본값은 0.8.
            
        Returns:
            List[SearchResult]: 중복이 제거된 검색 결과 목록
        """
        if not results or len(results) < 2:
            return results
            
        # 중복 쌍 탐지
        duplicates = self.detect_duplicates(results, threshold)
        
        if not duplicates:
            return results
            
        # 제거할 인덱스 집합
        to_remove = set()
        
        # 각 중복 쌍에서 점수가 낮은 항목 제거
        for i, j, _ in duplicates:
            if results[i].score >= results[j].score:
                to_remove.add(j)
            else:
                to_remove.add(i)
                
        # 중복이 제거된 결과 생성
        filtered_results = [result for idx, result in enumerate(results) if idx not in to_remove]
        
        logger.info(f"{len(results) - len(filtered_results)}개의 중복 결과 제거됨")
        return filtered_results
        
    def standardize_data(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        검색 결과 데이터 형식 표준화
        
        Args:
            results (List[SearchResult]): 검색 결과 목록
            
        Returns:
            List[SearchResult]: 표준화된 검색 결과 목록
        """
        if not results:
            return []
            
        standardized_results = []
        
        for result in results:
            # 제목 표준화
            title = result.title
            if not title or title == "제목 없음":
                # URL에서 제목 추출 시도
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(str(result.url))
                    path = parsed_url.path
                    if path and path != "/":
                        # 경로의 마지막 부분을 제목으로 사용
                        path_parts = path.strip("/").split("/")
                        if path_parts:
                            # 언더스코어와 하이픈을 공백으로 변환
                            title = path_parts[-1].replace("_", " ").replace("-", " ").capitalize()
                    
                    # 제목을 추출할 수 없는 경우 도메인 사용
                    if not title or title == "제목 없음":
                        title = parsed_url.netloc
                except Exception as e:
                    logger.warning(f"URL에서 제목 추출 실패: {str(e)}")
                    title = "제목 없음"
            
            # 내용 표준화
            content = self.clean_text(result.content)
            
            # 스니펫 표준화
            snippet = result.snippet
            if not snippet and content:
                # 내용에서 스니펫 생성
                snippet = content[:200] + ("..." if len(content) > 200 else "")
            
            # 메타데이터 표준화
            metadata = self.normalize_metadata(result.metadata)
            
            # 표준화된 결과 생성
            standardized_result = SearchResult(
                title=title,
                url=result.url,
                content=content,
                score=result.score,
                published_date=result.published_date,
                snippet=snippet,
                source_type=result.source_type,
                metadata=metadata
            )
            
            standardized_results.append(standardized_result)
            
        return standardized_results
        
    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        메타데이터 정규화
        
        Args:
            metadata (Dict[str, Any]): 원본 메타데이터
            
        Returns:
            Dict[str, Any]: 정규화된 메타데이터
        """
        if not metadata:
            return {}
            
        normalized = metadata.copy()
        
        # 검색 엔진 정규화
        if "search_engine" in normalized:
            engine = normalized["search_engine"].lower()
            if "tavily" in engine:
                normalized["search_engine"] = "tavily"
            elif "duckduckgo" in engine or "ddg" in engine:
                normalized["search_engine"] = "duckduckgo"
            elif "google" in engine:
                normalized["search_engine"] = "google"
            elif "bing" in engine:
                normalized["search_engine"] = "bing"
                
        # 도메인 정규화
        if "domain" in normalized:
            domain = normalized["domain"].lower()
            # www. 제거
            if domain.startswith("www."):
                domain = domain[4:]
            normalized["domain"] = domain
            
        # 타임스탬프 정규화
        if "timestamp" in normalized and isinstance(normalized["timestamp"], str):
            try:
                # ISO 형식 문자열을 datetime으로 변환
                timestamp = datetime.fromisoformat(normalized["timestamp"].replace('Z', '+00:00'))
                # 다시 ISO 형식 문자열로 변환 (표준화)
                normalized["timestamp"] = timestamp.isoformat()
            except (ValueError, TypeError):
                pass
                
        # 점수 정규화 (0.0 ~ 1.0 범위로)
        if "score" in normalized and isinstance(normalized["score"], (int, float)):
            score = float(normalized["score"])
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
            normalized["score"] = score
            
        return normalized
        
    def evaluate_domain_reliability(self, domain: str) -> float:
        """
        도메인 신뢰도 평가
        
        Args:
            domain (str): 평가할 도메인
            
        Returns:
            float: 신뢰도 점수 (0.0 ~ 1.0)
        """
        if not domain:
            return 0.5  # 기본값
            
        # 도메인 정규화
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        # 미리 정의된 도메인 점수 확인
        if domain in self.domain_scores:
            return self.domain_scores[domain]
            
        # 신뢰할 수 있는 도메인 확인
        for trusted_domain in self.trusted_domains:
            if trusted_domain in domain:
                return 0.8  # 신뢰할 수 있는 도메인은 기본적으로 높은 점수
                
        # 도메인 TLD 기반 점수
        if domain.endswith(".edu"):
            return 0.85
        elif domain.endswith(".gov"):
            return 0.9
        elif domain.endswith(".org"):
            return 0.75
        elif domain.endswith(".com"):
            return 0.6
        elif domain.endswith(".net"):
            return 0.6
        elif domain.endswith(".io"):
            return 0.65
        
        # 기타 도메인
        return 0.5  # 기본값
        
    def evaluate_content_quality(self, result: SearchResult) -> float:
        """
        콘텐츠 품질 평가
        
        Args:
            result (SearchResult): 평가할 검색 결과
            
        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        if not result or not result.content:
            return 0.0
            
        score = 0.5  # 기본 점수
        
        # 도메인 신뢰도 반영
        domain = ""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(str(result.url))
            domain = parsed_url.netloc
        except:
            if "domain" in result.metadata:
                domain = result.metadata["domain"]
                
        if domain:
            domain_score = self.evaluate_domain_reliability(domain)
            score = score * 0.3 + domain_score * 0.7  # 도메인 신뢰도 가중치 70%
            
        # 콘텐츠 길이 반영
        content_length = len(result.content)
        if content_length < 100:
            score *= 0.7  # 너무 짧은 콘텐츠는 감점
        elif content_length > 1000:
            score *= 1.2  # 긴 콘텐츠는 가산점 (최대 1.0)
            
        # 발행 날짜 반영
        if result.published_date:
            # 최근 콘텐츠일수록 높은 점수
            days_old = (datetime.now() - result.published_date).days
            if days_old < 30:  # 한 달 이내
                score *= 1.2
            elif days_old < 365:  # 1년 이내
                score *= 1.1
            elif days_old > 1825:  # 5년 이상
                score *= 0.8
                
        # 점수 범위 제한
        return max(0.0, min(1.0, score))
        
    def merge_search_results(self, tavily_results: List[SearchResult], duckduckgo_results: List[SearchResult]) -> List[SearchResult]:
        """
        Tavily와 DuckDuckGo 검색 결과 통합
        
        Args:
            tavily_results (List[SearchResult]): Tavily 검색 결과 목록
            duckduckgo_results (List[SearchResult]): DuckDuckGo 검색 결과 목록
            
        Returns:
            List[SearchResult]: 통합된 검색 결과 목록
        """
        if not tavily_results and not duckduckgo_results:
            return []
            
        # 결과가 하나만 있는 경우
        if not tavily_results:
            return duckduckgo_results
        if not duckduckgo_results:
            return tavily_results
            
        # 모든 결과 합치기
        all_results = tavily_results + duckduckgo_results
        
        # 중복 제거
        filtered_results = self.filter_duplicates(all_results)
        
        # 점수 기준 정렬
        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)
        
        return sorted_results
        
    def adjust_result_rankings(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        검색 결과 순위 조정
        
        콘텐츠 품질과 도메인 신뢰도를 고려하여 검색 결과 순위를 조정합니다.
        
        Args:
            results (List[SearchResult]): 검색 결과 목록
            
        Returns:
            List[SearchResult]: 순위가 조정된 검색 결과 목록
        """
        if not results:
            return []
            
        # 각 결과에 대해 품질 점수 계산
        scored_results = []
        for result in results:
            # 콘텐츠 품질 평가
            quality_score = self.evaluate_content_quality(result)
            
            # 원래 점수와 품질 점수를 결합
            adjusted_score = result.score * 0.6 + quality_score * 0.4
            
            # 조정된 점수로 새 SearchResult 객체 생성
            adjusted_result = SearchResult(
                title=result.title,
                url=result.url,
                content=result.content,
                score=adjusted_score,  # 조정된 점수
                published_date=result.published_date,
                snippet=result.snippet,
                source_type=result.source_type,
                metadata=result.metadata
            )
            
            scored_results.append(adjusted_result)
            
        # 조정된 점수 기준으로 정렬
        sorted_results = sorted(scored_results, key=lambda x: x.score, reverse=True)
        
        return sorted_results