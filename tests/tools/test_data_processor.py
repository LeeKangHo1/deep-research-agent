# tests/tools/test_data_processor.py

"""
DataProcessor 클래스 테스트
"""

import unittest
from datetime import datetime
from typing import List

from src.tools.data_processor import DataProcessor
from src.models.search import SearchResult


class TestDataProcessor(unittest.TestCase):
    """DataProcessor 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.processor = DataProcessor(use_nlp=False)
        
        # 테스트용 검색 결과 생성
        self.test_results = [
            SearchResult(
                title="파이썬 프로그래밍 기초",
                url="https://example.com/python-basics",
                content="<p>파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.</p><p>초보자도 쉽게 배울 수 있습니다.</p>",
                score=0.9,
                published_date=datetime(2023, 1, 15),
                snippet="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.",
                source_type="webpage",
                metadata={"search_engine": "tavily", "domain": "example.com"}
            ),
            SearchResult(
                title="파이썬 기초 학습하기",
                url="https://another-site.com/learn-python",
                content="<div>파이썬은 간결한 문법을 가진 언어로, 초보자에게 적합합니다.</div><div>다양한 라이브러리를 제공합니다.</div>",
                score=0.8,
                published_date=datetime(2023, 2, 20),
                snippet="파이썬은 간결한 문법을 가진 언어로, 초보자에게 적합합니다.",
                source_type="webpage",
                metadata={"search_engine": "duckduckgo", "domain": "another-site.com"}
            ),
            SearchResult(
                title="자바스크립트 입문",
                url="https://example.com/javascript-intro",
                content="<p>자바스크립트는 웹 개발에 필수적인 언어입니다.</p><p>브라우저에서 실행되는 스크립트 언어입니다.</p>",
                score=0.7,
                published_date=datetime(2023, 3, 10),
                snippet="자바스크립트는 웹 개발에 필수적인 언어입니다.",
                source_type="webpage",
                metadata={"search_engine": "tavily", "domain": "example.com"}
            )
        ]
        
    def test_clean_text(self):
        """텍스트 정제 테스트"""
        html_text = "<p>이것은 <b>HTML</b> 태그가 포함된 텍스트입니다.</p>"
        expected = "이것은 HTML 태그가 포함된 텍스트입니다."
        
        result = self.processor.clean_text(html_text)
        self.assertEqual(result, expected)
        
    def test_remove_html_tags(self):
        """HTML 태그 제거 테스트"""
        html_text = "<div>이것은 <span style='color:red'>HTML</span> 태그가 포함된 텍스트입니다.</div>"
        expected = "이것은 HTML 태그가 포함된 텍스트입니다."
        
        result = self.processor.remove_html_tags(html_text)
        self.assertEqual(result, expected)
        
    def test_clean_special_chars(self):
        """특수 문자 정리 테스트"""
        text = "이것은!!!   특수   문자가...   포함된   텍스트입니다."
        expected = "이것은! 특수 문자가... 포함된 텍스트입니다."
        
        result = self.processor.clean_special_chars(text)
        self.assertEqual(result, expected)
        
    def test_detect_duplicates(self):
        """중복 탐지 테스트"""
        # 중복 결과 추가
        duplicate_result = SearchResult(
            title="파이썬 프로그래밍 기초",
            url="https://different-url.com/python",
            content="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다. 초보자도 쉽게 배울 수 있습니다.",
            score=0.85,
            published_date=datetime(2023, 4, 5),
            snippet="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.",
            source_type="webpage",
            metadata={"search_engine": "duckduckgo", "domain": "different-url.com"}
        )
        
        test_results = self.test_results + [duplicate_result]
        
        duplicates = self.processor.detect_duplicates(test_results)
        
        # 첫 번째 결과와 마지막 결과가 중복으로 탐지되어야 함
        self.assertTrue(len(duplicates) > 0)
        
        # 중복 쌍 확인
        found_duplicate = False
        for i, j, similarity in duplicates:
            if (i == 0 and j == 3) or (i == 3 and j == 0):
                found_duplicate = True
                self.assertGreaterEqual(similarity, 0.8)  # 유사도가 임계값 이상이어야 함
                
        self.assertTrue(found_duplicate)
        
    def test_filter_duplicates(self):
        """중복 필터링 테스트"""
        # 중복 결과 추가
        duplicate_result = SearchResult(
            title="파이썬 프로그래밍 기초",
            url="https://different-url.com/python",
            content="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다. 초보자도 쉽게 배울 수 있습니다.",
            score=0.85,
            published_date=datetime(2023, 4, 5),
            snippet="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.",
            source_type="webpage",
            metadata={"search_engine": "duckduckgo", "domain": "different-url.com"}
        )
        
        test_results = self.test_results + [duplicate_result]
        
        filtered_results = self.processor.filter_duplicates(test_results)
        
        # 중복이 제거되어 원래 결과 수보다 적어야 함
        self.assertLess(len(filtered_results), len(test_results))
        
        # 점수가 더 높은 결과가 유지되어야 함
        found_original = False
        for result in filtered_results:
            if result.title == "파이썬 프로그래밍 기초":
                found_original = True
                self.assertEqual(result.score, 0.9)  # 원래 결과의 점수
                
        self.assertTrue(found_original)
        
    def test_standardize_data(self):
        """데이터 표준화 테스트"""
        # 표준화가 필요한 결과 생성
        non_standard_result = SearchResult(
            title="",  # 빈 제목
            url="https://example.org/article/python-guide",
            content="<script>alert('test');</script>이것은 정제가 필요한 <b>콘텐츠</b>입니다.",
            score=0.75,
            published_date=None,
            snippet=None,  # 빈 스니펫
            source_type=None,
            metadata={"search_engine": "TAVILY", "domain": "www.example.org"}  # 대문자 및 www 포함
        )
        
        standardized = self.processor.standardize_data([non_standard_result])[0]
        
        # 제목이 URL에서 추출되었는지 확인
        self.assertNotEqual(standardized.title, "")
        # URL의 도메인이나 경로에서 추출된 제목인지 확인
        self.assertTrue("example.org" in standardized.title.lower() or "python" in standardized.title.lower())
        
        # 콘텐츠가 정제되었는지 확인
        self.assertNotIn("<script>", standardized.content)
        self.assertNotIn("<b>", standardized.content)
        
        # 스니펫이 생성되었는지 확인
        self.assertIsNotNone(standardized.snippet)
        
        # 메타데이터가 정규화되었는지 확인
        self.assertEqual(standardized.metadata["search_engine"], "tavily")
        self.assertEqual(standardized.metadata["domain"], "example.org")
        
    def test_normalize_metadata(self):
        """메타데이터 정규화 테스트"""
        metadata = {
            "search_engine": "TAVILY-API",
            "domain": "www.Example.com",
            "score": 1.5,  # 범위 초과
            "timestamp": "2023-05-15T12:34:56Z"
        }
        
        normalized = self.processor.normalize_metadata(metadata)
        
        self.assertEqual(normalized["search_engine"], "tavily")
        self.assertEqual(normalized["domain"], "example.com")
        self.assertEqual(normalized["score"], 1.0)  # 1.0으로 제한됨
        self.assertTrue("timestamp" in normalized)
        
    def test_evaluate_domain_reliability(self):
        """도메인 신뢰도 평가 테스트"""
        # 신뢰할 수 있는 도메인
        self.assertGreaterEqual(self.processor.evaluate_domain_reliability("wikipedia.org"), 0.8)
        self.assertGreaterEqual(self.processor.evaluate_domain_reliability("github.com"), 0.8)
        
        # 교육 및 정부 도메인
        self.assertGreaterEqual(self.processor.evaluate_domain_reliability("stanford.edu"), 0.8)
        self.assertGreaterEqual(self.processor.evaluate_domain_reliability("nasa.gov"), 0.8)
        
        # 일반 도메인
        self.assertLessEqual(self.processor.evaluate_domain_reliability("example.com"), 0.7)
        
    def test_evaluate_content_quality(self):
        """콘텐츠 품질 평가 테스트"""
        # 높은 품질의 결과
        high_quality = SearchResult(
            title="파이썬 고급 프로그래밍",
            url="https://wikipedia.org/wiki/Python",
            content="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다. " * 20,  # 긴 콘텐츠
            score=0.9,
            published_date=datetime.now(),  # 최신 콘텐츠
            snippet="파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.",
            source_type="webpage",
            metadata={"domain": "wikipedia.org"}  # 신뢰할 수 있는 도메인
        )
        
        # 낮은 품질의 결과
        low_quality = SearchResult(
            title="파이썬 팁",
            url="https://unknown-site.com/tips",
            content="파이썬 팁입니다.",  # 짧은 콘텐츠
            score=0.5,
            published_date=datetime(2010, 1, 1),  # 오래된 콘텐츠
            snippet="파이썬 팁입니다.",
            source_type="webpage",
            metadata={"domain": "unknown-site.com"}  # 알려지지 않은 도메인
        )
        
        high_score = self.processor.evaluate_content_quality(high_quality)
        low_score = self.processor.evaluate_content_quality(low_quality)
        
        self.assertGreater(high_score, low_score)
        
    def test_merge_search_results(self):
        """검색 결과 통합 테스트"""
        tavily_results = [self.test_results[0], self.test_results[2]]  # 첫 번째와 세 번째 결과
        duckduckgo_results = [self.test_results[1]]  # 두 번째 결과
        
        merged = self.processor.merge_search_results(tavily_results, duckduckgo_results)
        
        # 모든 결과가 포함되어야 함
        self.assertEqual(len(merged), 3)
        
        # 점수 순으로 정렬되어야 함
        self.assertEqual(merged[0].score, 0.9)
        self.assertEqual(merged[1].score, 0.8)
        self.assertEqual(merged[2].score, 0.7)
        
    def test_adjust_result_rankings(self):
        """결과 순위 조정 테스트"""
        # 원래 순위: 0.9 > 0.8 > 0.7
        adjusted = self.processor.adjust_result_rankings(self.test_results)
        
        # 순위가 조정되었는지 확인
        self.assertEqual(len(adjusted), 3)
        
        # 점수가 조정되었는지 확인
        for i, result in enumerate(adjusted):
            self.assertNotEqual(result.score, self.test_results[i].score)
            
        # 여전히 점수 순으로 정렬되어 있어야 함
        for i in range(len(adjusted) - 1):
            self.assertGreaterEqual(adjusted[i].score, adjusted[i + 1].score)


if __name__ == "__main__":
    unittest.main()