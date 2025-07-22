# src/utils/tests/test_context_window.py

"""
컨텍스트 윈도우 관리 기능 테스트
"""

import unittest
from datetime import datetime, timedelta
from src.utils.conversation_memory import ConversationMemory
from src.models.conversation import ConversationMessage


class TestContextWindowManagement(unittest.TestCase):
    """컨텍스트 윈도우 관리 기능 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.memory = ConversationMemory(context_window_size=10, max_memory_size=100)
        
        # 테스트용 메시지 생성
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            # 메시지 길이를 다양하게 설정하여 복잡성 테스트
            content_length = 50 if i < 10 else 200
            content = f"메시지 {i+1} " + "테스트 내용 " * (content_length // 10)
            
            # 시간 간격을 두고 메시지 추가
            timestamp = datetime.now() - timedelta(minutes=(20-i)*5)
            
            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=timestamp
            )
            self.memory.history.add_message(message)
    
    def test_basic_context_window(self):
        """기본 컨텍스트 윈도우 테스트"""
        # 기본 크기(10)로 컨텍스트 윈도우 가져오기
        context = self.memory.get_context_window(dynamic_size=False)
        
        # 최근 10개 메시지만 반환되는지 확인
        self.assertEqual(len(context), 10)
        
        # 최근 메시지가 맞는지 확인
        self.assertEqual(context[0].content.startswith("메시지 11"), True)
        self.assertEqual(context[-1].content.startswith("메시지 20"), True)
    
    def test_dynamic_context_window(self):
        """동적 컨텍스트 윈도우 테스트"""
        # 동적 크기로 컨텍스트 윈도우 가져오기
        context = self.memory.get_context_window(dynamic_size=True)
        
        # 동적 크기는 기본 크기(10)보다 크거나 같아야 함
        self.assertGreaterEqual(len(context), 10)
        
        # 최대 메모리 크기(100)보다 작거나 같아야 함
        self.assertLessEqual(len(context), 100)
    
    def test_window_size_adjustment(self):
        """윈도우 크기 조정 테스트"""
        # 윈도우 크기 변경
        self.memory.set_context_window_size(15)
        
        # 변경된 크기로 컨텍스트 윈도우 가져오기
        context = self.memory.get_context_window(dynamic_size=False)
        
        # 변경된 크기(15)로 메시지가 반환되는지 확인
        self.assertEqual(len(context), 15)
        
        # 최근 메시지가 맞는지 확인
        self.assertEqual(context[0].content.startswith("메시지 6"), True)
        self.assertEqual(context[-1].content.startswith("메시지 20"), True)
    
    def test_adaptive_context_window(self):
        """적응형 컨텍스트 윈도우 테스트"""
        # 대화 복잡성에 따른 최적 크기 계산
        optimal_size, context = self.memory.optimize_context_window()
        
        # 최적 크기가 계산되었는지 확인
        self.assertGreater(optimal_size, 0)
        
        # 컨텍스트 윈도우 크기가 최적 크기와 일치하는지 확인
        self.assertEqual(len(context), optimal_size)
    
    def test_context_window_with_token_limit(self):
        """토큰 제한이 있는 컨텍스트 윈도우 테스트"""
        # 토큰 제한으로 적응형 컨텍스트 가져오기
        context = self.memory.get_adaptive_context(max_tokens=1000)
        
        # 컨텍스트가 반환되었는지 확인
        self.assertGreater(len(context), 0)
        
        # 시스템 메시지 추가
        self.memory.add_message("system", "시스템 지시사항입니다.")
        
        # 시스템 메시지가 포함된 컨텍스트 가져오기
        context_with_system = self.memory.get_adaptive_context(max_tokens=1000)
        
        # 시스템 메시지가 포함되었는지 확인
        system_messages = [msg for msg in context_with_system if msg.role == "system"]
        self.assertGreater(len(system_messages), 0)
    
    def test_context_with_summary(self):
        """요약이 포함된 컨텍스트 테스트"""
        # 요약이 포함된 컨텍스트 가져오기
        context = self.memory.get_context_with_summary()
        
        # 컨텍스트가 반환되었는지 확인
        self.assertGreater(len(context), 0)
        
        # 첫 번째 메시지가 시스템 메시지(요약)인지 확인
        self.assertEqual(context[0]["role"], "system")
        self.assertIn("이전 대화 요약", context[0]["content"])


if __name__ == "__main__":
    unittest.main()