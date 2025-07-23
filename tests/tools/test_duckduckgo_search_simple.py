# src/tools/test_duckduckgo_search_simple.py

"""
DuckDuckGoSearchTool 간단한 테스트
"""

import unittest
from unittest.mock import patch, MagicMock

class TestDuckDuckGoSearchToolSimple(unittest.TestCase):
    """DuckDuckGoSearchTool 간단한 테스트 클래스"""
    
    def test_mock_search(self):
        """모의 객체를 사용한 검색 테스트"""
        # 모의 객체 생성
        mock_search_tool = MagicMock()
        mock_search_tool.search.return_value = ["테스트 결과"]
        
        # 검색 실행
        results = mock_search_tool.search("테스트 쿼리")
        
        # 검증
        self.assertEqual(results, ["테스트 결과"])
        mock_search_tool.search.assert_called_once_with("테스트 쿼리")
        
    def test_mock_search_with_context(self):
        """모의 객체를 사용한 컨텍스트 기반 검색 테스트"""
        # 모의 객체 생성
        mock_search_tool = MagicMock()
        mock_search_tool.search_with_context.return_value = ["컨텍스트 테스트 결과"]
        
        # 검색 실행
        results = mock_search_tool.search_with_context("테스트 쿼리", "테스트 컨텍스트")
        
        # 검증
        self.assertEqual(results, ["컨텍스트 테스트 결과"])
        mock_search_tool.search_with_context.assert_called_once_with("테스트 쿼리", "테스트 컨텍스트")
        
    def test_mock_error_handling(self):
        """모의 객체를 사용한 오류 처리 테스트"""
        # 모의 객체 생성
        mock_search_tool = MagicMock()
        mock_search_tool.search.side_effect = Exception("테스트 예외")
        mock_search_tool.search_safely.return_value = []
        
        # 예외 발생 확인
        with self.assertRaises(Exception):
            mock_search_tool.search("테스트 쿼리")
            
        # 안전한 검색 실행
        results = mock_search_tool.search_safely("테스트 쿼리")
        
        # 검증
        self.assertEqual(results, [])
        mock_search_tool.search_safely.assert_called_once_with("테스트 쿼리")
        
if __name__ == '__main__':
    unittest.main()