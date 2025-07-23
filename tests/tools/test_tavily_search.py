# src/tools/test_tavily_search.py

"""
TavilySearchTool 테스트
Tavily API 호출을 최소화하기 위해 모든 테스트에서 모킹을 사용합니다.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json
from datetime import datetime
import requests

from src.tools.tavily_search import (
    TavilySearchTool, 
    TavilySearchException,
    TavilyAPIKeyError,
    TavilyConnectionError,
    TavilyResponseError,
    TavilyRateLimitError,
    TavilyTimeoutError
)
from src.models.search import SearchResult

class TestTavilySearchTool(unittest.TestCase):
    """TavilySearchTool 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 환경 변수 설정
        os.environ["TAVILY_API_KEY"] = "tvly-test-api-key"
        
        # 테스트용 검색 결과
        self.mock_search_response = {
            "query": "파이썬 프로그래밍",
            "search_depth": "basic",
            "results": [
                {
                    "title": "파이썬 프로그래밍 기초",
                    "url": "https://example.com/python-basics",
                    "content": "파이썬은 초보자에게 적합한 프로그래밍 언어입니다.",
                    "snippet": "파이썬은 초보자에게 적합한 프로그래밍 언어입니다.",
                    "source_type": "webpage",
                    "published_date": "2023-01-01T00:00:00Z"
                },
                {
                    "title": "파이썬 고급 기법",
                    "url": "https://example.com/python-advanced",
                    "content": "파이썬의 고급 기법에 대해 알아봅니다.",
                    "snippet": "파이썬의 고급 기법에 대해 알아봅니다.",
                    "source_type": "webpage",
                    "published_date": "2023-02-01T00:00:00Z"
                }
            ]
        }
        
        # 테스트용 컨텍스트 검색 결과
        self.mock_context_search_response = {
            "query": "파이썬 데이터 분석",
            "search_depth": "basic",
            "results": [
                {
                    "title": "파이썬 판다스 기초",
                    "url": "https://example.com/python-pandas",
                    "content": "판다스는 파이썬에서 데이터 분석을 위한 강력한 라이브러리입니다.",
                    "snippet": "판다스는 파이썬에서 데이터 분석을 위한 강력한 라이브러리입니다.",
                    "source_type": "webpage",
                    "published_date": "2023-03-01T00:00:00Z"
                },
                {
                    "title": "파이썬 넘파이 튜토리얼",
                    "url": "https://example.com/python-numpy",
                    "content": "넘파이는 파이썬에서 수치 계산을 위한 기본 라이브러리입니다.",
                    "snippet": "넘파이는 파이썬에서 수치 계산을 위한 기본 라이브러리입니다.",
                    "source_type": "webpage",
                    "published_date": "2023-04-01T00:00:00Z"
                }
            ]
        }
    
    def tearDown(self):
        """테스트 정리"""
        # 환경 변수 제거
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_initialization_and_basic_search(self, mock_tavily_client):
        """초기화 및 기본 검색 테스트"""
        # 모의 객체 설정
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = self.mock_search_response
        mock_tavily_client.return_value = mock_client_instance
        
        # 테스트 대상 객체 생성
        search_tool = TavilySearchTool()
        
        # 초기화 검증
        self.assertIsNotNone(search_tool.client)
        self.assertEqual(search_tool.api_key, "tvly-test-api-key")
        mock_tavily_client.assert_called_once_with(api_key="tvly-test-api-key")
        
        # 기본 검색 수행
        results = search_tool.search("파이썬 프로그래밍")
        
        # 검색 결과 검증
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].title, "파이썬 프로그래밍 기초")
        # URL은 문자열로 변환하여 비교
        self.assertEqual(str(results[0].url), "https://example.com/python-basics")
        self.assertEqual(results[0].content, "파이썬은 초보자에게 적합한 프로그래밍 언어입니다.")
        
        # 검색 호출 검증
        mock_client_instance.search.assert_called_once_with(query="파이썬 프로그래밍", search_depth="basic")
        
        # 최대 결과 수 제한 테스트
        mock_client_instance.search.reset_mock()
        mock_client_instance.search.return_value = self.mock_search_response
        
        results = search_tool.search("파이썬 프로그래밍", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "파이썬 프로그래밍 기초")
        
        # 검색 깊이 설정 테스트
        mock_client_instance.search.reset_mock()
        mock_client_instance.search.return_value = self.mock_search_response
        
        results = search_tool.search("파이썬 프로그래밍", search_depth="advanced")
        mock_client_instance.search.assert_called_once_with(query="파이썬 프로그래밍", search_depth="advanced")
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_api_key_validation(self, mock_tavily_client):
        """API 키 검증 테스트"""
        # 환경 변수 제거
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        
        # API 키 없이 초기화 시도
        with self.assertRaises(TavilyAPIKeyError):
            TavilySearchTool()
        
        # 유효하지 않은 API 키로 초기화 시도
        os.environ["TAVILY_API_KEY"] = "invalid-api-key"
        with self.assertRaises(TavilyAPIKeyError):
            TavilySearchTool()
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_context_based_search(self, mock_tavily_client):
        """컨텍스트 기반 검색 테스트"""
        # 모의 객체 설정
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = self.mock_context_search_response
        mock_tavily_client.return_value = mock_client_instance
        
        # 테스트 대상 객체 생성
        search_tool = TavilySearchTool()
        
        # 컨텍스트 기반 검색 수행
        context = "데이터 분석 프로젝트를 위한 파이썬 라이브러리를 찾고 있습니다."
        results = search_tool.search_with_context("파이썬 데이터 분석", context)
        
        # 검색 결과 검증
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].title, "파이썬 판다스 기초")
        # URL은 문자열로 변환하여 비교
        self.assertEqual(str(results[0].url), "https://example.com/python-pandas")
        
        # 검색 호출 검증
        mock_client_instance.search.assert_called_once()
        call_args = mock_client_instance.search.call_args[1]
        self.assertEqual(call_args["search_depth"], "basic")
        self.assertTrue("파이썬 데이터 분석" in call_args["query"])
        
        # 컨텍스트 가중치 테스트
        mock_client_instance.search.reset_mock()
        mock_client_instance.search.return_value = self.mock_context_search_response
        
        results = search_tool.search_with_context("파이썬 데이터 분석", context, context_weight=0.8)
        self.assertEqual(len(results), 2)
        mock_client_instance.search.assert_called_once()
        
        # 빈 컨텍스트 테스트
        mock_client_instance.search.reset_mock()
        mock_client_instance.search.return_value = self.mock_search_response
        
        results = search_tool.search_with_context("파이썬 프로그래밍", "")
        mock_client_instance.search.assert_called_once()
        call_args = mock_client_instance.search.call_args[1]
        self.assertEqual(call_args["query"], "파이썬 프로그래밍")
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_error_handling(self, mock_tavily_client):
        """오류 처리 테스트"""
        # 모의 객체 설정
        mock_client_instance = MagicMock()
        mock_tavily_client.return_value = mock_client_instance
        
        # 테스트 대상 객체 생성
        search_tool = TavilySearchTool()
        
        # 연결 오류 테스트
        mock_client_instance.search.side_effect = requests.exceptions.ConnectionError("연결 오류")
        with self.assertRaises(TavilyConnectionError):
            # _handle_api_error 메서드를 직접 호출하여 테스트
            search_tool._handle_api_error(requests.exceptions.ConnectionError("연결 오류"), "파이썬 프로그래밍")
        
        # 타임아웃 오류 테스트
        mock_client_instance.search.side_effect = requests.exceptions.Timeout("타임아웃")
        with self.assertRaises(TavilyTimeoutError):
            # _handle_api_error 메서드를 직접 호출하여 테스트
            search_tool._handle_api_error(requests.exceptions.Timeout("타임아웃"), "파이썬 프로그래밍")
        
        # 유효하지 않은 응답 테스트
        mock_client_instance.search.return_value = None
        with self.assertRaises(TavilySearchException):
            # 일반적인 예외는 TavilySearchException으로 변환됨
            search_tool._handle_api_error(ValueError("유효하지 않은 응답"), "파이썬 프로그래밍")
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_retry_mechanism(self, mock_tavily_client):
        """재시도 메커니즘 테스트"""
        # 모의 객체 설정
        mock_client_instance = MagicMock()
        mock_tavily_client.return_value = mock_client_instance
        
        # 테스트 대상 객체 생성
        search_tool = TavilySearchTool()
        
        # 재시도 후 성공 테스트
        mock_client_instance.search.side_effect = [
            requests.exceptions.ConnectionError("연결 오류"),
            self.mock_search_response
        ]
        
        # search 메서드를 패치하여 재시도 메커니즘 테스트
        with patch.object(search_tool, 'search', side_effect=[
            TavilyConnectionError("연결 오류"),
            self.mock_search_response
        ]) as mock_search:
            results = search_tool.search_with_retry("파이썬 프로그래밍", retry_delay=0.01)
            self.assertEqual(mock_search.call_count, 2)
        
        # 최대 재시도 횟수 초과 테스트
        with patch.object(search_tool, 'search', side_effect=TavilyConnectionError("연결 오류")) as mock_search:
            with self.assertRaises(TavilyConnectionError):
                search_tool.search_with_retry("파이썬 프로그래밍", max_retries=1, retry_delay=0.01)
            self.assertEqual(mock_search.call_count, 2)  # 초기 시도 1회 + 재시도 1회
    
    @patch('src.tools.tavily_search.TavilyClient')
    def test_result_conversion(self, mock_tavily_client):
        """검색 결과 변환 테스트"""
        # 모의 객체 설정
        mock_tavily_client.return_value = MagicMock()
        
        # 원시 검색 결과
        raw_result = {
            "title": "파이썬 프로그래밍 기초",
            "url": "https://example.com/python-basics",
            "content": "파이썬은 초보자에게 적합한 프로그래밍 언어입니다.",
            "snippet": "파이썬은 초보자에게 적합한 프로그래밍 언어입니다.",
            "source_type": "webpage",
            "published_date": "2023-01-01T00:00:00Z"
        }
        
        # SearchResult 객체로 변환
        result = TavilySearchTool.convert_to_search_result(raw_result, "파이썬 프로그래밍", 1)
        
        # 검증
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.title, "파이썬 프로그래밍 기초")
        # URL은 문자열로 변환하여 비교
        self.assertEqual(str(result.url), "https://example.com/python-basics")
        self.assertEqual(result.metadata["query"], "파이썬 프로그래밍")
        self.assertEqual(result.metadata["rank"], 1)
        
        # 원시 검색 결과 목록 변환
        raw_results = [
            raw_result,
            {
                "title": "파이썬 고급 기법",
                "url": "https://example.com/python-advanced",
                "content": "파이썬의 고급 기법에 대해 알아봅니다.",
                "snippet": "파이썬의 고급 기법에 대해 알아봅니다.",
                "source_type": "webpage",
                "published_date": "2023-02-01T00:00:00Z"
            }
        ]
        
        results = TavilySearchTool.convert_raw_results(raw_results, "파이썬 프로그래밍")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "파이썬 프로그래밍 기초")
        self.assertEqual(results[1].title, "파이썬 고급 기법")

if __name__ == '__main__':
    unittest.main()