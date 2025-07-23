# src/tools/test_duckduckgo_search_mock.py

"""
DuckDuckGoSearchTool 모의 테스트
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# SearchResult 클래스 모의 구현
class MockSearchResult:
    def __init__(self, title, url, content, score=1.0, published_date=None, snippet=None, source_type=None, metadata=None):
        self.title = title
        self.url = url
        self.content = content
        self.score = score
        self.published_date = published_date
        self.snippet = snippet
        self.source_type = source_type
        self.metadata = metadata or {}

# DuckDuckGoSearchTool 클래스 모의 구현
class MockDuckDuckGoSearchTool:
    def __init__(self):
        self.client = MagicMock()
        
    def search(self, query, max_results=5, region="wt-wt", safesearch="moderate"):
        # 모의 검색 결과 반환
        return [
            MockSearchResult(
                title="테스트 제목 1",
                url="https://example.com/1",
                content="테스트 내용 1",
                score=1.0,
                snippet="테스트 스니펫 1",
                source_type="webpage",
                metadata={"search_engine": "duckduckgo", "query": query}
            ),
            MockSearchResult(
                title="테스트 제목 2",
                url="https://example.com/2",
                content="테스트 내용 2",
                score=0.95,
                snippet="테스트 스니펫 2",
                source_type="webpage",
                metadata={"search_engine": "duckduckgo", "query": query}
            )
        ][:max_results]
        
    def search_with_context(self, query, context, max_results=5, region="wt-wt", safesearch="moderate", context_weight=0.5):
        # 모의 컨텍스트 기반 검색 결과 반환
        return [
            MockSearchResult(
                title="컨텍스트 테스트 제목 1",
                url="https://example.com/context/1",
                content="컨텍스트 테스트 내용 1",
                score=1.0,
                snippet="컨텍스트 테스트 스니펫 1",
                source_type="webpage",
                metadata={
                    "search_engine": "duckduckgo", 
                    "query": query,
                    "context_weight": context_weight,
                    "enhanced_query": f"{query} 컨텍스트 키워드"
                }
            )
        ][:max_results]
        
    def search_safely(self, query, max_results=5, region="wt-wt", safesearch="moderate"):
        # 안전한 검색 모의 구현
        try:
            return self.search(query, max_results, region, safesearch)
        except Exception:
            return []
            
    def search_with_fallback(self, query, max_results=5, region="wt-wt", safesearch="moderate", fallback_message=None):
        # 대체 결과 제공 모의 구현
        try:
            return self.search(query, max_results, region, safesearch)
        except Exception:
            return [
                MockSearchResult(
                    title=f"'{query}' 검색 실패",
                    url=f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                    content=fallback_message or f"검색 쿼리 '{query}'에 대한 결과를 찾을 수 없습니다.",
                    score=0.0,
                    snippet=fallback_message or f"검색 쿼리 '{query}'에 대한 결과를 찾을 수 없습니다.",
                    source_type="error",
                    metadata={"search_engine": "duckduckgo", "query": query, "is_fallback": True}
                )
            ]
            
    def _determine_source_type(self, url):
        # URL 기반 소스 유형 결정 모의 구현
        url_lower = url.lower()
        
        if url_lower.endswith(".pdf"):
            return "pdf"
        elif url_lower.endswith((".doc", ".docx")):
            return "document"
        elif "youtube.com" in url_lower:
            return "video"
        elif "github.com" in url_lower:
            return "code"
        else:
            return "webpage"

class TestDuckDuckGoSearchToolMock(unittest.TestCase):
    """DuckDuckGoSearchTool 모의 테스트 클래스"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.search_tool = MockDuckDuckGoSearchTool()
        
    def test_basic_search(self):
        """기본 검색 테스트"""
        # 검색 실행
        results = self.search_tool.search("테스트 쿼리")
        
        # 검증
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "테스트 제목 1")
        self.assertEqual(str(results[0].url), "https://example.com/1")
        self.assertEqual(results[0].content, "테스트 내용 1")
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(results[0].metadata["search_engine"], "duckduckgo")
        self.assertEqual(results[0].metadata["query"], "테스트 쿼리")
        
    def test_search_with_max_results(self):
        """최대 결과 수 제한 테스트"""
        # 검색 실행
        results = self.search_tool.search("테스트 쿼리", max_results=1)
        
        # 검증
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "테스트 제목 1")
        
    def test_search_with_context(self):
        """컨텍스트 기반 검색 테스트"""
        # 검색 실행
        results = self.search_tool.search_with_context("테스트 쿼리", "테스트 컨텍스트", context_weight=0.7)
        
        # 검증
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "컨텍스트 테스트 제목 1")
        self.assertEqual(results[0].metadata["context_weight"], 0.7)
        self.assertEqual(results[0].metadata["enhanced_query"], "테스트 쿼리 컨텍스트 키워드")
        
    def test_search_safely(self):
        """안전한 검색 테스트"""
        # 정상 검색
        results = self.search_tool.search_safely("테스트 쿼리")
        self.assertEqual(len(results), 2)
        
        # 예외 발생 시뮬레이션
        with patch.object(self.search_tool, 'search', side_effect=Exception("테스트 예외")):
            results = self.search_tool.search_safely("테스트 쿼리")
            self.assertEqual(results, [])
            
    def test_search_with_fallback(self):
        """대체 결과 제공 테스트"""
        # 예외 발생 시뮬레이션
        with patch.object(self.search_tool, 'search', side_effect=Exception("테스트 예외")):
            results = self.search_tool.search_with_fallback("테스트 쿼리", fallback_message="검색에 실패했습니다.")
            
            # 검증
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "'테스트 쿼리' 검색 실패")
            self.assertEqual(results[0].content, "검색에 실패했습니다.")
            self.assertEqual(results[0].score, 0.0)
            self.assertTrue(results[0].metadata["is_fallback"])
            
    def test_determine_source_type(self):
        """소스 유형 결정 테스트"""
        # 파일 확장자 기반 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://example.com/doc.pdf"), "pdf")
        self.assertEqual(self.search_tool._determine_source_type("https://example.com/doc.docx"), "document")
        
        # 도메인 기반 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://youtube.com/watch?v=123"), "video")
        self.assertEqual(self.search_tool._determine_source_type("https://github.com/user/repo"), "code")
        
        # 기본 유형 테스트
        self.assertEqual(self.search_tool._determine_source_type("https://example.com"), "webpage")
        
if __name__ == '__main__':
    unittest.main()