# src/tools/test_duckduckgo_search.py

"""
DuckDuckGoSearchTool 테스트
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime

from src.tools.duckduckgo_search import (
    DuckDuckGoSearchTool,
    DuckDuckGoSearchException,
    DuckDuckGoConnectionError,
    DuckDuckGoResponseError,
    DuckDuckGoRateLimitError,
    DuckDuckGoTimeoutError
)
from src.models.search import SearchResult

class TestDuckDuckGoSearchTool(unittest.TestCase):
    """DuckDuckGoSearchTool 테스트 클래스"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        # 실제 API 호출을 방지하기 위해 클라이언트 초기화 메서드를 모킹
        with patch('src.tools.duckduckgo_search.DDGS') as mock_ddgs:
            self.mock_client = MagicMock()
            mock_ddgs.return_value = self.mock_client
            self.search_tool = DuckDuckGoSearchTool()
            
        # 테스트용 검색 결과 설정
        self.test_results = [
            {"title": "테스트 제목 1", "href": "https://example.com/1", "body": "테스트 내용 1"},
            {"title": "테스트 제목 2", "href": "https://example.com/2", "body": "테스트 내용 2"}
        ]
            
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.search_tool)
        self.assertIsNotNone(self.search_tool.client)
        
    def test_is_client_available(self):
        """클라이언트 사용 가능 여부 확인 테스트"""
        self.assertTrue(self.search_tool.is_client_available())
        
        # 클라이언트가 None인 경우
        self.search_tool.client = None
        self.assertFalse(self.search_tool.is_client_available())
        
    @patch('src.tools.duckduckgo_search.DDGS')
    def test_initialize_safely(self, mock_ddgs):
        """안전한 초기화 테스트"""
        # 정상 초기화
        mock_ddgs.return_value = MagicMock()
        search_tool = DuckDuckGoSearchTool.initialize_safely()
        self.assertIsNotNone(search_tool)
        self.assertIsNotNone(search_tool.client)
        
        # 예외 발생 시 제한된 기능으로 초기화
        mock_ddgs.side_effect = Exception("테스트 예외")
        search_tool = DuckDuckGoSearchTool.initialize_safely()
        self.assertIsNotNone(search_tool)
        self.assertIsNone(search_tool.client)
        
    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results')
    def test_basic_search(self, mock_parse):
        """기본 검색 테스트"""
        # 모의 검색 결과 설정
        mock_results = [
            {"title": "테스트 제목 1", "href": "https://example.com/1", "body": "테스트 내용 1"},
            {"title": "테스트 제목 2", "href": "https://example.com/2", "body": "테스트 내용 2"}
        ]
        self.mock_client.text.return_value = mock_results
        
        # 모의 파싱 결과 설정
        expected_results = [
            SearchResult(
                title="테스트 제목 1",
                url="https://example.com/1",
                content="테스트 내용 1",
                score=1.0
            ),
            SearchResult(
                title="테스트 제목 2",
                url="https://example.com/2",
                content="테스트 내용 2",
                score=0.95
            )
        ]
        mock_parse.return_value = expected_results
        
        # 검색 실행
        results = self.search_tool.search("테스트 쿼리")
        
        # 검증
        self.mock_client.text.assert_called_once()
        mock_parse.assert_called_once()
        self.assertEqual(results, expected_results)
        
    def test_search_with_parameters(self):
        """파라미터를 사용한 검색 테스트"""
        # 모의 검색 결과 설정
        mock_results = [{"title": "테스트", "href": "https://example.com", "body": "내용"}]
        self.mock_client.text.return_value = mock_results
        
        # 검색 실행
        self.search_tool.search("테스트 쿼리", max_results=10, region="kr-ko", safesearch="off")
        
        # 검증
        self.mock_client.text.assert_called_once_with(
            "테스트 쿼리", 
            region="kr-ko", 
            safesearch="off", 
            max_results=10
        )
        
    def test_search_with_invalid_parameters(self):
        """유효하지 않은 파라미터를 사용한 검색 테스트"""
        # 모의 검색 결과 설정
        mock_results = [{"title": "테스트", "href": "https://example.com", "body": "내용"}]
        self.mock_client.text.return_value = mock_results
        
        # 검색 실행 (유효하지 않은 max_results)
        self.search_tool.search("테스트 쿼리", max_results=-1, safesearch="invalid")
        
        # 검증 (기본값으로 대체되어야 함)
        self.mock_client.text.assert_called_once_with(
            "테스트 쿼리", 
            region="wt-wt", 
            safesearch="moderate", 
            max_results=5
        )
        
    def test_search_with_none_response(self):
        """None 응답을 반환하는 검색 테스트"""
        # 모의 검색 결과 설정
        self.mock_client.text.return_value = None
        
        # 검색 실행 및 예외 발생 확인
        with self.assertRaises(DuckDuckGoResponseError):
            self.search_tool.search("테스트 쿼리")
            
    def test_search_with_empty_response(self):
        """빈 응답을 반환하는 검색 테스트"""
        # 모의 검색 결과 설정
        self.mock_client.text.return_value = []
        
        # 검색 실행
        results = self.search_tool.search("테스트 쿼리")
        
        # 검증 (빈 결과 목록이 반환되어야 함)
        self.assertEqual(results, [])
        
    def test_search_with_exception(self):
        """예외가 발생하는 검색 테스트"""
        # 모의 예외 설정
        self.mock_client.text.side_effect = Exception("테스트 예외")
        
        # 검색 실행 및 예외 발생 확인
        with self.assertRaises(DuckDuckGoSearchException):
            self.search_tool.search("테스트 쿼리")
            
    def test_search_safely(self):
        """안전한 검색 테스트"""
        # 모의 검색 결과 설정
        mock_results = [{"title": "테스트", "href": "https://example.com", "body": "내용"}]
        self.mock_client.text.return_value = mock_results
        
        # 검색 실행
        results = self.search_tool.search_safely("테스트 쿼리")
        
        # 검증
        self.mock_client.text.assert_called_once()
        self.assertEqual(len(results), 1)  # 결과가 있어야 함
        
        # 예외 발생 시 빈 결과 반환
        self.mock_client.text.side_effect = Exception("테스트 예외")
        results = self.search_tool.search_safely("테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록이 반환되어야 함
        
    def test_search_with_fallback(self):
        """대체 결과를 사용한 검색 테스트"""
        # 모의 예외 설정
        self.mock_client.text.side_effect = Exception("테스트 예외")
        
        # 검색 실행
        results = self.search_tool.search_with_fallback("테스트 쿼리")
        
        # 검증 (대체 결과가 반환되어야 함)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "'테스트 쿼리' 검색 실패")
        self.assertTrue(results[0].metadata.get("is_fallback", False))
        
    def test_determine_source_type(self):
        """소스 유형 결정 테스트"""
        # 파일 확장자 기반 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://example.com/doc.pdf"), "pdf")
        self.assertEqual(self.search_tool._determine_source_type("https://example.com/doc.docx"), "document")
        self.assertEqual(self.search_tool._determine_source_type("https://example.com/image.jpg"), "image")
        
        # 도메인 기반 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://youtube.com/watch?v=123"), "video")
        self.assertEqual(self.search_tool._determine_source_type("https://github.com/user/repo"), "code")
        self.assertEqual(self.search_tool._determine_source_type("https://wikipedia.org/wiki/Page"), "encyclopedia")
        
        # 기본 유형 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://example.com"), "webpage")
        
    def test_parse_search_results(self):
        """검색 결과 파싱 테스트"""
        # 모의 검색 결과
        raw_results = [
            {
                "title": "테스트 제목 1",
                "href": "https://example.com/1",
                "body": "테스트 내용 1"
            },
            {
                "title": "테스트 제목 2",
                "href": "https://example.com/2",
                "body": "테스트 내용 2"
            },
            {
                "title": "",  # 제목 없음
                "href": "https://example.com/3",
                "body": "테스트 내용 3"
            },
            {
                "title": "테스트 제목 4",
                "href": "",  # URL 없음
                "body": "테스트 내용 4"
            },
            {
                "title": "테스트 제목 5",
                "href": "https://example.com/5",
                "body": ""  # 내용 없음
            }
        ]
        
        # 파싱 실행
        results = self.search_tool._parse_search_results(raw_results, "테스트 쿼리")
        
        # 검증
        self.assertEqual(len(results), 3)  # URL이 없는 결과는 제외되어야 함
        
        # 첫 번째 결과 검증
        self.assertEqual(results[0].title, "테스트 제목 1")
        self.assertEqual(str(results[0].url), "https://example.com/1")
        self.assertEqual(results[0].content, "테스트 내용 1")
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(results[0].metadata["search_engine"], "duckduckgo")
        self.assertEqual(results[0].metadata["query"], "테스트 쿼리")
        
        # 두 번째 결과 검증
        self.assertEqual(results[1].title, "테스트 제목 2")
        self.assertEqual(str(results[1].url), "https://example.com/2")
        self.assertEqual(results[1].content, "테스트 내용 2")
        self.assertEqual(results[1].score, 0.95)
        
        # 세 번째 결과 검증 (제목 없음)
        self.assertNotEqual(results[2].title, "")  # 제목이 URL에서 추출되어야 함
        
    def test_convert_to_search_result(self):
        """단일 검색 결과 변환 테스트"""
        # 모의 검색 결과
        raw_result = {
            "title": "테스트 제목",
            "href": "https://example.com",
            "body": "테스트 내용",
            "date": "2023-01-01"
        }
        
        # 변환 실행
        result = DuckDuckGoSearchTool.convert_to_search_result(raw_result, "테스트 쿼리", 1)
        
        # 검증
        self.assertEqual(result.title, "테스트 제목")
        self.assertEqual(str(result.url), "https://example.com")
        self.assertEqual(result.content, "테스트 내용")
        self.assertEqual(result.score, 0.95)  # 두 번째 결과이므로 0.95
        self.assertEqual(result.metadata["search_engine"], "duckduckgo")
        self.assertEqual(result.metadata["query"], "테스트 쿼리")
        
        # URL이 없는 경우
        raw_result = {"title": "테스트", "body": "내용"}
        result = DuckDuckGoSearchTool.convert_to_search_result(raw_result)
        self.assertIsNone(result)  # URL이 없으면 None 반환
        
    def test_convert_raw_results(self):
        """여러 검색 결과 변환 테스트"""
        # 모의 검색 결과
        raw_results = [
            {"title": "테스트 1", "href": "https://example.com/1", "body": "내용 1"},
            {"title": "테스트 2", "href": "https://example.com/2", "body": "내용 2"},
            {"title": "테스트 3", "body": "내용 3"}  # URL 없음
        ]
        
        # 변환 실행
        results = DuckDuckGoSearchTool.convert_raw_results(raw_results, "테스트 쿼리")
        
        # 검증
        self.assertEqual(len(results), 2)  # URL이 없는 결과는 제외되어야 함
        self.assertEqual(results[0].title, "테스트 1")
        self.assertEqual(results[1].title, "테스트 2")
        
    def test_merge_search_results(self):
        """검색 결과 병합 테스트"""
        # 모의 검색 결과
        results1 = [
            SearchResult(title="결과 1", url="https://example.com/1", content="내용 1", score=1.0),
            SearchResult(title="결과 2", url="https://example.com/2", content="내용 2", score=0.9)
        ]
        
        results2 = [
            SearchResult(title="결과 3", url="https://example.com/3", content="내용 3", score=0.95),
            SearchResult(title="결과 2", url="https://example.com/2", content="중복 내용", score=0.8)  # 중복 URL
        ]
        
        # 병합 실행
        merged = self.search_tool.merge_search_results([results1, results2], max_results=3)
        
        # 검증
        self.assertEqual(len(merged), 3)  # 중복 제거 후 3개 결과
        self.assertEqual(merged[0].title, "결과 1")  # 점수 기준 정렬
        self.assertEqual(merged[1].title, "결과 3")
        self.assertEqual(merged[2].title, "결과 2")
        
        # 최대 결과 수 제한 테스트
        merged = self.search_tool.merge_search_results([results1, results2], max_results=2)
        self.assertEqual(len(merged), 2)  # 최대 2개로 제한

    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._enhance_query_with_context')
    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results')
    def test_search_with_context(self, mock_parse, mock_enhance):
        """컨텍스트 기반 검색 테스트"""
        # 모의 향상된 쿼리 설정
        mock_enhance.return_value = "테스트 쿼리 향상된 컨텍스트"
        
        # 모의 검색 결과 설정
        self.mock_client.text.return_value = self.test_results
        
        # 모의 파싱 결과 설정
        expected_results = [
            SearchResult(
                title="테스트 제목 1",
                url="https://example.com/1",
                content="테스트 내용 1",
                score=1.0
            ),
            SearchResult(
                title="테스트 제목 2",
                url="https://example.com/2",
                content="테스트 내용 2",
                score=0.95
            )
        ]
        mock_parse.return_value = expected_results
        
        # 컨텍스트 기반 검색 실행
        results = self.search_tool.search_with_context(
            "테스트 쿼리", 
            "테스트 컨텍스트", 
            max_results=5, 
            context_weight=0.7
        )
        
        # 검증
        mock_enhance.assert_called_once_with("테스트 쿼리", "테스트 컨텍스트", 0.7)
        self.mock_client.text.assert_called_once_with(
            "테스트 쿼리 향상된 컨텍스트", 
            region="wt-wt", 
            safesearch="moderate", 
            max_results=5
        )
        mock_parse.assert_called_once()
        self.assertEqual(results, expected_results)
        
    def test_enhance_query_with_context(self):
        """컨텍스트를 사용한 쿼리 향상 테스트"""
        # 컨텍스트 가중치가 0인 경우
        enhanced = self.search_tool._enhance_query_with_context("테스트 쿼리", "테스트 컨텍스트", 0.0)
        self.assertEqual(enhanced, "테스트 쿼리")  # 원본 쿼리가 반환되어야 함
        
        # 컨텍스트가 없는 경우
        enhanced = self.search_tool._enhance_query_with_context("테스트 쿼리", "", 0.5)
        self.assertEqual(enhanced, "테스트 쿼리")  # 원본 쿼리가 반환되어야 함
        
        # 일반적인 경우
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._extract_keywords_from_context') as mock_extract:
            mock_extract.return_value = "키워드1 키워드2 키워드3"
            
            # 가중치 0.5 (일부 키워드만 사용)
            enhanced = self.search_tool._enhance_query_with_context("테스트 쿼리", "테스트 컨텍스트", 0.5)
            self.assertEqual(enhanced, "테스트 쿼리 키워드1")  # 첫 번째 키워드만 사용
            
            # 가중치 1.0 (모든 키워드 사용)
            enhanced = self.search_tool._enhance_query_with_context("테스트 쿼리", "테스트 컨텍스트", 1.0)
            self.assertEqual(enhanced, "테스트 쿼리 키워드1 키워드2 키워드3")  # 모든 키워드 사용
            
    def test_extract_keywords_from_context(self):
        """컨텍스트에서 키워드 추출 테스트"""
        # 간단한 영어 컨텍스트
        keywords = self.search_tool._extract_keywords_from_context("This is a test context with important keywords")
        self.assertTrue(len(keywords) > 0)
        self.assertIn("test", keywords)
        self.assertIn("context", keywords)
        self.assertIn("important", keywords)
        self.assertIn("keywords", keywords)
        
        # 불용어가 제외되어야 함
        self.assertNotIn("this", keywords.lower())
        self.assertNotIn("is", keywords.lower())
        self.assertNotIn("a", keywords.lower())
        self.assertNotIn("with", keywords.lower())
        
        # 한국어 컨텍스트
        keywords = self.search_tool._extract_keywords_from_context("이것은 중요한 키워드가 포함된 테스트 컨텍스트입니다")
        self.assertTrue(len(keywords) > 0)
        self.assertIn("중요한", keywords)
        self.assertIn("키워드", keywords)
        self.assertIn("테스트", keywords)
        self.assertIn("컨텍스트", keywords)
        
        # 한국어 불용어가 제외되어야 함
        self.assertNotIn("이것은", keywords)
        self.assertNotIn("포함된", keywords)
        self.assertNotIn("입니다", keywords)
        
    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._enhance_query_with_multiple_contexts')
    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results')
    def test_search_with_weighted_context(self, mock_parse, mock_enhance):
        """가중치 컨텍스트 기반 검색 테스트"""
        # 모의 향상된 쿼리 설정
        mock_enhance.return_value = "테스트 쿼리 다중 컨텍스트"
        
        # 모의 검색 결과 설정
        self.mock_client.text.return_value = self.test_results
        
        # 모의 파싱 결과 설정
        expected_results = [
            SearchResult(
                title="테스트 제목 1",
                url="https://example.com/1",
                content="테스트 내용 1",
                score=1.0
            )
        ]
        mock_parse.return_value = expected_results
        
        # 컨텍스트 목록 설정
        contexts = [
            {"text": "첫 번째 컨텍스트", "weight": 0.8, "type": "conversation"},
            {"text": "두 번째 컨텍스트", "weight": 0.5, "type": "document"}
        ]
        
        # 가중치 컨텍스트 기반 검색 실행
        results = self.search_tool.search_with_weighted_context(
            "테스트 쿼리", 
            contexts, 
            max_results=3
        )
        
        # 검증
        mock_enhance.assert_called_once_with("테스트 쿼리", contexts)
        self.mock_client.text.assert_called_once_with(
            "테스트 쿼리 다중 컨텍스트", 
            region="wt-wt", 
            safesearch="moderate", 
            max_results=3
        )
        mock_parse.assert_called_once()
        self.assertEqual(results, expected_results)
        
    def test_enhance_query_with_multiple_contexts(self):
        """여러 컨텍스트를 사용한 쿼리 향상 테스트"""
        # 컨텍스트가 없는 경우
        enhanced = self.search_tool._enhance_query_with_multiple_contexts("테스트 쿼리", [])
        self.assertEqual(enhanced, "테스트 쿼리")  # 원본 쿼리가 반환되어야 함
        
        # 여러 컨텍스트가 있는 경우
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._extract_keywords_from_context') as mock_extract:
            # 첫 번째 컨텍스트에서 추출된 키워드
            mock_extract.side_effect = ["키워드1 키워드2", "키워드3 키워드4 키워드5"]
            
            contexts = [
                {"text": "첫 번째 컨텍스트", "weight": 0.5},
                {"text": "두 번째 컨텍스트", "weight": 1.0}
            ]
            
            enhanced = self.search_tool._enhance_query_with_multiple_contexts("테스트 쿼리", contexts)
            
            # 중복 제거 및 가중치 적용 후 키워드가 포함되어야 함
            self.assertIn("테스트 쿼리", enhanced)
            for keyword in ["키워드1", "키워드3", "키워드4", "키워드5"]:
                self.assertIn(keyword, enhanced)
                
    def test_adjust_context_weight(self):
        """컨텍스트 가중치 조정 테스트"""
        # 컨텍스트 유형에 따른 조정
        weight = self.search_tool.adjust_context_weight(0.5, "conversation", 100)
        self.assertGreater(weight, 0.5)  # 대화 컨텍스트는 가중치 증가
        
        weight = self.search_tool.adjust_context_weight(0.5, "user_profile", 100)
        self.assertLess(weight, 0.5)  # 사용자 프로필은 가중치 감소
        
        # 컨텍스트 길이에 따른 조정
        weight = self.search_tool.adjust_context_weight(0.5, "document", 10)
        self.assertLess(weight, 0.5)  # 매우 짧은 컨텍스트는 가중치 감소
        
        weight = self.search_tool.adjust_context_weight(0.5, "document", 100)
        self.assertGreaterEqual(weight, 0.5)  # 적당한 길이의 컨텍스트는 가중치 유지 또는 증가
        
        weight = self.search_tool.adjust_context_weight(0.5, "document", 600)
        self.assertLess(weight, 0.5)  # 매우 긴 컨텍스트는 가중치 감소
        
        # 컨텍스트 나이에 따른 조정
        weight = self.search_tool.adjust_context_weight(0.5, "document", 100, 30)
        self.assertGreater(weight, 0.5)  # 최근 컨텍스트는 가중치 증가
        
        weight = self.search_tool.adjust_context_weight(0.5, "document", 100, 7200)
        self.assertLess(weight, 0.5)  # 오래된 컨텍스트는 가중치 감소
        
    @patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results')
    def test_search_with_context_safely(self, mock_parse):
        """안전한 컨텍스트 기반 검색 테스트"""
        # 모의 검색 결과 설정
        self.mock_client.text.return_value = self.test_results
        
        # 모의 파싱 결과 설정
        expected_results = [
            SearchResult(
                title="테스트 제목 1",
                url="https://example.com/1",
                content="테스트 내용 1",
                score=1.0
            )
        ]
        mock_parse.return_value = expected_results
        
        # 안전한 컨텍스트 기반 검색 실행
        results = self.search_tool.search_with_context_safely(
            "테스트 쿼리", 
            "테스트 컨텍스트"
        )
        
        # 검증
        self.mock_client.text.assert_called_once()
        mock_parse.assert_called_once()
        self.assertEqual(results, expected_results)
        
        # 예외 발생 시 빈 결과 반환
        self.mock_client.text.side_effect = Exception("테스트 예외")
        results = self.search_tool.search_with_context_safely("테스트 쿼리", "테스트 컨텍스트")
        self.assertEqual(results, [])

    def test_handle_api_error(self):
        """API 오류 처리 테스트"""
        # 속도 제한 오류
        rate_limit_error = Exception("Rate limit exceeded")
        with self.assertRaises(DuckDuckGoRateLimitError):
            self.search_tool._handle_api_error(rate_limit_error, "테스트 쿼리")
            
        # 타임아웃 오류
        timeout_error = Exception("Request timed out")
        with self.assertRaises(DuckDuckGoTimeoutError):
            self.search_tool._handle_api_error(timeout_error, "테스트 쿼리")
            
        # 연결 오류
        connection_error = Exception("Connection refused")
        with self.assertRaises(DuckDuckGoConnectionError):
            self.search_tool._handle_api_error(connection_error, "테스트 쿼리")
            
        # 응답 오류
        response_error = Exception("Bad response: 500 Internal Server Error")
        with self.assertRaises(DuckDuckGoResponseError):
            self.search_tool._handle_api_error(response_error, "테스트 쿼리")
            
        # 기타 오류
        other_error = Exception("Unknown error")
        with self.assertRaises(DuckDuckGoSearchException):
            self.search_tool._handle_api_error(other_error, "테스트 쿼리")
            
    def test_handle_search_error(self):
        """검색 오류 처리 테스트"""
        # 속도 제한 오류
        rate_limit_error = DuckDuckGoRateLimitError("Rate limit exceeded", retry_after=60)
        results = self.search_tool.handle_search_error(rate_limit_error, "테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록 반환
        
        # 타임아웃 오류
        timeout_error = DuckDuckGoTimeoutError("Request timed out")
        results = self.search_tool.handle_search_error(timeout_error, "테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록 반환
        
        # 연결 오류
        connection_error = DuckDuckGoConnectionError("Connection refused")
        results = self.search_tool.handle_search_error(connection_error, "테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록 반환
        
        # 응답 오류
        response_error = DuckDuckGoResponseError("Bad response", status_code=500)
        results = self.search_tool.handle_search_error(response_error, "테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록 반환
        
        # 기타 오류
        other_error = Exception("Unknown error")
        results = self.search_tool.handle_search_error(other_error, "테스트 쿼리")
        self.assertEqual(results, [])  # 빈 결과 목록 반환
        
    def test_search_with_retry(self):
        """자동 재시도 기능 테스트"""
        # 첫 번째 시도에서 오류 발생, 두 번째 시도에서 성공
        self.mock_client.text.side_effect = [
            Exception("Connection error"),  # 첫 번째 시도: 실패
            self.test_results  # 두 번째 시도: 성공
        ]
        
        # 모의 파싱 결과 설정
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results') as mock_parse:
            expected_results = [
                SearchResult(
                    title="테스트 제목 1",
                    url="https://example.com/1",
                    content="테스트 내용 1",
                    score=1.0
                )
            ]
            mock_parse.return_value = expected_results
            
            # 재시도 기능으로 검색 실행
            results = self.search_tool.search_with_retry(
                "테스트 쿼리", 
                max_retries=3, 
                retry_delay=0.01  # 테스트 속도를 위해 짧은 지연 시간 사용
            )
            
            # 검증
            self.assertEqual(self.mock_client.text.call_count, 2)  # 두 번 호출되어야 함
            self.assertEqual(results, expected_results)
            
        # 모든 시도 실패
        self.mock_client.text.reset_mock()
        self.mock_client.text.side_effect = Exception("Persistent error")
        
        # 재시도 기능으로 검색 실행
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._create_fallback_result') as mock_fallback:
            mock_fallback.return_value = ["대체 결과"]
            
            results = self.search_tool.search_with_retry(
                "테스트 쿼리", 
                max_retries=2, 
                retry_delay=0.01
            )
            
            # 검증
            self.assertEqual(self.mock_client.text.call_count, 3)  # 세 번 호출되어야 함 (이전 2번 + 새로운 3번)
            mock_fallback.assert_called_once()
            self.assertEqual(results, ["대체 결과"])
            
    def test_search_with_fallback(self):
        """대체 결과 제공 기능 테스트"""
        # 검색 실패 시뮬레이션
        self.mock_client.text.side_effect = Exception("Search failed")
        
        # 대체 결과 생성 모킹
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._create_fallback_result') as mock_fallback:
            mock_fallback.return_value = ["대체 결과"]
            
            # 대체 결과 제공 기능으로 검색 실행
            results = self.search_tool.search_with_fallback(
                "테스트 쿼리", 
                fallback_message="검색에 실패했습니다."
            )
            
            # 검증
            self.mock_client.text.assert_called_once()
            mock_fallback.assert_called_once_with("테스트 쿼리", "검색에 실패했습니다.", error="Search failed")
            self.assertEqual(results, ["대체 결과"])
            
    def test_create_fallback_result(self):
        """대체 결과 생성 테스트"""
        # 기본 대체 결과
        results = self.search_tool._create_fallback_result("테스트 쿼리")
        
        # 검증
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "'테스트 쿼리' 검색 실패")
        self.assertTrue(isinstance(results[0], SearchResult))
        self.assertEqual(results[0].score, 0.0)  # 대체 결과는 점수가 0
        self.assertEqual(results[0].metadata["search_engine"], "duckduckgo")
        self.assertTrue(results[0].metadata["is_fallback"])
        
        # 사용자 지정 메시지와 오류 정보가 있는 대체 결과
        results = self.search_tool._create_fallback_result(
            "테스트 쿼리", 
            fallback_message="사용자 지정 메시지", 
            error="테스트 오류"
        )
        
        # 검증
        self.assertEqual(len(results), 1)
        self.assertIn("사용자 지정 메시지", results[0].content)
        self.assertIn("테스트 오류", results[0].content)
        self.assertEqual(results[0].metadata["error"], "테스트 오류")
        
    def test_search_with_backup_strategy(self):
        """백업 전략 기능 테스트"""
        # 첫 번째 전략 실패, 두 번째 전략 성공
        self.mock_client.text.side_effect = [
            Exception("First strategy failed"),  # 첫 번째 전략: 실패
            [],  # 두 번째 전략: 결과 없음
            self.test_results  # 세 번째 전략: 성공
        ]
        
        # 모의 파싱 결과 설정
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._parse_search_results') as mock_parse:
            mock_parse.return_value = [
                SearchResult(
                    title="테스트 제목 1",
                    url="https://example.com/1",
                    content="테스트 내용 1",
                    score=1.0
                )
            ]
            
            # 백업 전략 설정
            backup_strategies = [
                {"region": "us-en"},  # 두 번째 전략
                {"modified_query": "테스트 쿼리 수정됨"}  # 세 번째 전략
            ]
            
            # 백업 전략 기능으로 검색 실행
            results = self.search_tool.search_with_backup_strategy(
                "테스트 쿼리", 
                backup_strategies=backup_strategies
            )
            
            # 검증
            self.assertEqual(self.mock_client.text.call_count, 3)  # 세 번 호출되어야 함
            self.assertEqual(len(results), 1)
            
            # 세 번째 호출 파라미터 검증
            call_args = self.mock_client.text.call_args_list[2][0]
            self.assertEqual(call_args[0], "테스트 쿼리 수정됨")  # 수정된 쿼리 사용
            
        # 모든 전략 실패
        self.mock_client.text.reset_mock()
        self.mock_client.text.side_effect = [
            Exception("First strategy failed"),
            Exception("Second strategy failed"),
            []  # 결과 없음
        ]
        
        # 대체 결과 생성 모킹
        with patch('src.tools.duckduckgo_search.DuckDuckGoSearchTool._create_fallback_result') as mock_fallback:
            mock_fallback.return_value = ["대체 결과"]
            
            # 백업 전략 기능으로 검색 실행
            results = self.search_tool.search_with_backup_strategy(
                "테스트 쿼리", 
                backup_strategies=backup_strategies
            )
            
            # 검증
            self.assertEqual(self.mock_client.text.call_count, 3)  # 세 번 호출되어야 함
            mock_fallback.assert_called_once()
            self.assertEqual(results, ["대체 결과"])
            
if __name__ == '__main__':
    unittest.main()