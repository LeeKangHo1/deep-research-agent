# src/models/test_search_models.py

"""
검색 관련 데이터 모델 테스트
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from .search import SearchResult


class TestSearchResult(unittest.TestCase):
    """SearchResult 클래스 테스트"""
    
    def test_create_search_result(self):
        """기본 SearchResult 객체 생성 테스트"""
        result = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다.",
            score=0.85
        )
        
        self.assertEqual(result.title, "테스트 제목")
        self.assertEqual(str(result.url), "https://example.com/")
        self.assertEqual(result.content, "테스트 내용입니다.")
        self.assertEqual(result.score, 0.85)
        self.assertIsNone(result.published_date)
        
    def test_create_search_result_with_optional_fields(self):
        """선택적 필드를 포함한 SearchResult 객체 생성 테스트"""
        now = datetime.now()
        result = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다.",
            score=0.85,
            published_date=now,
            snippet="테스트 스니펫",
            source_type="웹페이지",
            metadata={"category": "기술", "language": "ko"}
        )
        
        self.assertEqual(result.title, "테스트 제목")
        self.assertEqual(str(result.url), "https://example.com/")
        self.assertEqual(result.content, "테스트 내용입니다.")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.published_date, now)
        self.assertEqual(result.snippet, "테스트 스니펫")
        self.assertEqual(result.source_type, "웹페이지")
        self.assertEqual(result.metadata, {"category": "기술", "language": "ko"})
        
    def test_invalid_score(self):
        """유효하지 않은 점수로 SearchResult 객체 생성 시도"""
        with self.assertRaises(ValidationError):
            SearchResult(
                title="테스트 제목",
                url="https://example.com",
                content="테스트 내용입니다.",
                score=1.5  # 1.0보다 큰 값
            )
            
        with self.assertRaises(ValidationError):
            SearchResult(
                title="테스트 제목",
                url="https://example.com",
                content="테스트 내용입니다.",
                score=-0.5  # 0.0보다 작은 값
            )
            
    def test_invalid_url(self):
        """유효하지 않은 URL로 SearchResult 객체 생성 시도"""
        with self.assertRaises(ValidationError):
            SearchResult(
                title="테스트 제목",
                url="invalid-url",  # http:// 또는 https://로 시작하지 않음
                content="테스트 내용입니다.",
                score=0.85
            )
            
    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        now = datetime.now()
        result = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다.",
            score=0.85,
            published_date=now,
            snippet="테스트 스니펫"
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["title"], "테스트 제목")
        self.assertEqual(result_dict["url"], "https://example.com/")
        self.assertEqual(result_dict["content"], "테스트 내용입니다.")
        self.assertEqual(result_dict["score"], 0.85)
        self.assertEqual(result_dict["published_date"], now.isoformat())
        self.assertEqual(result_dict["snippet"], "테스트 스니펫")
        
    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        now = datetime.now()
        data = {
            "title": "테스트 제목",
            "url": "https://example.com",
            "content": "테스트 내용입니다.",
            "score": 0.85,
            "published_date": now.isoformat(),
            "snippet": "테스트 스니펫"
        }
        
        result = SearchResult.from_dict(data)
        self.assertEqual(result.title, "테스트 제목")
        self.assertEqual(str(result.url), "https://example.com/")
        self.assertEqual(result.content, "테스트 내용입니다.")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.published_date.isoformat(), now.isoformat())
        self.assertEqual(result.snippet, "테스트 스니펫")
        
    def test_get_summary(self):
        """get_summary 메서드 테스트"""
        # 스니펫이 있는 경우
        result1 = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다. 이 내용은 스니펫보다 길지만 요약에는 사용되지 않습니다.",
            score=0.85,
            snippet="이것은 스니펫입니다."
        )
        self.assertEqual(result1.get_summary(), "이것은 스니펫입니다.")
        
        # 스니펫이 없는 경우
        result2 = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다. 이 내용이 요약에 사용됩니다.",
            score=0.85
        )
        self.assertEqual(result2.get_summary(), "테스트 내용입니다. 이 내용이 요약에 사용됩니다.")
        
        # 최대 길이 제한 테스트
        long_content = "a" * 200
        result3 = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content=long_content,
            score=0.85
        )
        self.assertEqual(result3.get_summary(max_length=100), "a" * 100 + "...")
        
    def test_get_formatted_date(self):
        """get_formatted_date 메서드 테스트"""
        # 날짜가 있는 경우
        now = datetime(2023, 5, 15, 10, 30, 0)
        result1 = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다.",
            score=0.85,
            published_date=now
        )
        self.assertEqual(result1.get_formatted_date(), "2023-05-15")
        self.assertEqual(result1.get_formatted_date("%Y/%m/%d %H:%M"), "2023/05/15 10:30")
        
        # 날짜가 없는 경우
        result2 = SearchResult(
            title="테스트 제목",
            url="https://example.com",
            content="테스트 내용입니다.",
            score=0.85
        )
        self.assertIsNone(result2.get_formatted_date())


if __name__ == "__main__":
    unittest.main()