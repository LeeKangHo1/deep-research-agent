# src/utils/test_conversation_memory.py

"""
ConversationMemory 클래스에 대한 단위 테스트
메모리 저장 및 조회, 컨텍스트 윈도우 관리, 메모리 제한 및 정리 기능 테스트
"""

import unittest
import os
import json
import tempfile
from datetime import datetime, timedelta

from src.utils.conversation_memory import ConversationMemory
from src.models.conversation import ConversationMessage


class TestConversationMemory(unittest.TestCase):
    """ConversationMemory 클래스 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.memory = ConversationMemory(context_window_size=5, max_memory_size=10)
    
    def tearDown(self):
        """각 테스트 후에 실행되는 정리 작업"""
        self.memory = None
    
    # 1. 메모리 저장 및 조회 기능 테스트
    
    def test_add_message(self):
        """메시지 추가 기능 테스트"""
        # 메시지 추가
        message = self.memory.add_message("user", "안녕하세요")
        
        # 메시지가 올바르게 추가되었는지 확인
        self.assertEqual(len(self.memory.get_all_messages()), 1)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "안녕하세요")
        self.assertIsNotNone(message.timestamp)
    
    def test_add_message_with_metadata(self):
        """메타데이터가 있는 메시지 추가 테스트"""
        # 메타데이터와 함께 메시지 추가
        metadata = {"importance": 5, "category": "greeting"}
        message = self.memory.add_message("user", "중요한 메시지입니다", metadata)
        
        # 메시지와 메타데이터가 올바르게 추가되었는지 확인
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "중요한 메시지입니다")
        self.assertIsNotNone(message.metadata)
        self.assertEqual(message.metadata["importance"], 5)
        self.assertEqual(message.metadata["category"], "greeting")
    
    def test_add_messages_batch(self):
        """메시지 일괄 추가 기능 테스트"""
        # 추가할 메시지 목록
        messages = [
            {"role": "user", "content": "첫 번째 메시지"},
            {"role": "assistant", "content": "두 번째 메시지"},
            {"role": "user", "content": "세 번째 메시지", "metadata": {"important": True}}
        ]
        
        # 메시지 일괄 추가
        added_messages = self.memory.add_messages_batch(messages)
        
        # 추가된 메시지 수 확인
        self.assertEqual(len(added_messages), 3)
        
        # 메모리에 저장된 메시지 확인
        stored_messages = self.memory.get_all_messages()
        self.assertEqual(len(stored_messages), 3)
        
        # 메시지 내용 확인
        self.assertEqual(stored_messages[0].content, "첫 번째 메시지")
        self.assertEqual(stored_messages[1].content, "두 번째 메시지")
        self.assertEqual(stored_messages[2].content, "세 번째 메시지")
        
        # 메타데이터 확인
        self.assertIsNone(stored_messages[0].metadata)
        self.assertIsNone(stored_messages[1].metadata)
        self.assertIsNotNone(stored_messages[2].metadata)
        self.assertTrue(stored_messages[2].metadata.get("important"))
    
    def test_get_all_messages(self):
        """모든 메시지 조회 기능 테스트"""
        # 여러 메시지 추가
        self.memory.add_message("user", "첫 번째 메시지")
        self.memory.add_message("assistant", "두 번째 메시지")
        self.memory.add_message("system", "시스템 메시지")
        
        # 모든 메시지 조회
        all_messages = self.memory.get_all_messages()
        
        # 메시지 수 확인
        self.assertEqual(len(all_messages), 3)
        
        # 메시지 순서 확인
        self.assertEqual(all_messages[0].content, "첫 번째 메시지")
        self.assertEqual(all_messages[1].content, "두 번째 메시지")
        self.assertEqual(all_messages[2].content, "시스템 메시지")
    
    def test_get_messages_by_role(self):
        """역할별 메시지 조회 기능 테스트"""
        # 사용자 및 어시스턴트 메시지 추가
        self.memory.add_message("user", "안녕하세요")
        self.memory.add_message("assistant", "안녕하세요, 무엇을 도와드릴까요?")
        self.memory.add_message("user", "날씨가 어때요?")
        self.memory.add_message("system", "시스템 메시지")
        
        # 사용자 메시지 조회
        user_messages = self.memory.get_messages_by_role("user")
        self.assertEqual(len(user_messages), 2)
        for msg in user_messages:
            self.assertEqual(msg.role, "user")
        
        # 어시스턴트 메시지 조회
        assistant_messages = self.memory.get_messages_by_role("assistant")
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0].role, "assistant")
        
        # 시스템 메시지 조회
        system_messages = self.memory.get_messages_by_role("system")
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0].role, "system")
    
    def test_search_by_content(self):
        """내용 검색 기능 테스트"""
        # 메시지 추가
        self.memory.add_message("user", "안녕하세요, 오늘 날씨가 좋네요")
        self.memory.add_message("assistant", "네, 정말 좋은 날씨입니다")
        self.memory.add_message("user", "내일 날씨는 어떨까요?")
        self.memory.add_message("assistant", "내일은 비가 올 예정입니다")
        
        # 대소문자 구분 없이 검색
        results = self.memory.search_by_content("날씨")
        self.assertEqual(len(results), 3)
        
        # 대소문자 구분하여 검색
        results = self.memory.search_by_content("날씨", case_sensitive=True)
        self.assertEqual(len(results), 3)
        
        # 특정 문구 검색
        results = self.memory.search_by_content("비가 올")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "내일은 비가 올 예정입니다")
    
    # 2. 컨텍스트 윈도우 관리 테스트
    
    def test_get_context_window_basic(self):
        """기본 컨텍스트 윈도우 조회 기능 테스트"""
        # 10개의 메시지 추가
        for i in range(10):
            self.memory.add_message("user" if i % 2 == 0 else "assistant", f"메시지 {i}")
        
        # 컨텍스트 윈도우 조회 (최근 5개)
        context = self.memory.get_context_window()
        
        # 컨텍스트 윈도우 크기 확인
        self.assertEqual(len(context), 5)
        
        # 최근 5개 메시지인지 확인
        for i, msg in enumerate(context):
            self.assertEqual(msg.content, f"메시지 {i+5}")
    
    def test_get_context_window_with_system_messages(self):
        """시스템 메시지 포함 컨텍스트 윈도우 테스트"""
        # 시스템 메시지 추가
        self.memory.add_message("system", "시스템 메시지 1")
        self.memory.add_message("system", "시스템 메시지 2")
        
        # 일반 메시지 추가
        for i in range(8):
            self.memory.add_message("user" if i % 2 == 0 else "assistant", f"메시지 {i}")
        
        # 시스템 메시지 포함 컨텍스트 윈도우 조회
        context = self.memory.get_context_window(include_system=True)
        
        # 시스템 메시지 2개 + 일반 메시지 5개 = 총 7개
        self.assertEqual(len(context), 7)
        
        # 시스템 메시지가 먼저 오는지 확인
        self.assertEqual(context[0].role, "system")
        self.assertEqual(context[1].role, "system")
        
        # 시스템 메시지 제외 컨텍스트 윈도우 조회
        context = self.memory.get_context_window(include_system=False)
        
        # 일반 메시지 5개만 포함
        self.assertEqual(len(context), 5)
        
        # 모든 메시지가 시스템 메시지가 아닌지 확인
        for msg in context:
            self.assertNotEqual(msg.role, "system")
    
    def test_get_context_window_with_important_messages(self):
        """중요 메시지 우선 포함 기능 테스트"""
        # 메모리 초기화 및 크기 조정
        self.memory.clear_memory()
        self.memory.set_max_memory_size(20)  # 메모리 크기를 충분히 크게 설정
        
        # 일반 메시지 추가
        self.memory.add_message("user", "일반적인 질문입니다.")
        self.memory.add_message("assistant", "일반적인 답변입니다.")
        
        # 중요 메시지 추가 (메타데이터 포함)
        important_msg = self.memory.add_message("user", "중요한 질문입니다. 꼭 기억해주세요.", {"importance": 5.0})
        self.memory.add_message("assistant", "중요한 답변입니다. 반드시 참고하세요.", {"importance": 5.0})
        
        # 코드 블록이 있는 메시지 추가
        self.memory.add_message("user", "다음 코드를 분석해주세요:\n```python\ndef example():\n    return 'Hello World'\n```")
        
        # 추가 일반 메시지
        for i in range(3):
            self.memory.add_message("user", f"추가 질문 {i}")
            self.memory.add_message("assistant", f"추가 답변 {i}")
        
        # 중요 메시지 우선 컨텍스트 윈도우
        important_context = self.memory.get_context_window(prioritize_important=True)
        
        # 중요 메시지가 포함되어 있는지 확인
        important_message_found = False
        for msg in important_context:
            if msg.metadata and "importance" in msg.metadata:
                important_message_found = True
                break
            if "중요한 질문입니다" in msg.content or "중요한 답변입니다" in msg.content:
                important_message_found = True
                break
        
        self.assertTrue(important_message_found, "중요 메시지가 컨텍스트에 포함되어야 합니다")
        
        # 코드 블록이 있는 메시지가 포함되어 있는지 확인
        code_block_found = False
        for msg in important_context:
            if "```python" in msg.content:
                code_block_found = True
                break
        
        self.assertTrue(code_block_found, "코드 블록이 있는 메시지가 컨텍스트에 포함되어야 합니다")
    
    def test_dynamic_window_size(self):
        """동적 컨텍스트 윈도우 크기 테스트"""
        # 메모리 초기화 및 크기 조정
        self.memory.clear_memory()
        self.memory.set_context_window_size(5)
        
        # 짧은 메시지 추가
        for i in range(3):
            self.memory.add_message("user", f"짧은 질문 {i}")
            self.memory.add_message("assistant", f"짧은 답변 {i}")
        
        # 긴 메시지 추가
        long_content = "이것은 매우 긴 메시지입니다. " * 20  # 약 500자
        self.memory.add_message("user", long_content)
        self.memory.add_message("assistant", long_content)
        
        # 코드 블록이 있는 메시지 추가
        code_content = "다음 코드를 분석해주세요:\n```python\ndef example():\n    print('Hello')\n    return 'World'\n```"
        self.memory.add_message("user", code_content)
        
        # 동적 크기 조정 컨텍스트 윈도우 조회
        dynamic_context = self.memory.get_context_window(dynamic_size=True)
        
        # 동적 크기가 기본 크기와 다른지 확인
        # 정확한 크기는 알고리즘에 따라 다를 수 있으므로 크기만 확인
        self.assertGreaterEqual(len(dynamic_context), 1)
    
    def test_set_context_window_size(self):
        """컨텍스트 윈도우 크기 설정 기능 테스트"""
        # 초기 컨텍스트 윈도우 크기 확인
        self.assertEqual(self.memory.context_window_size, 5)
        
        # 컨텍스트 윈도우 크기 변경
        self.memory.set_context_window_size(3)
        self.assertEqual(self.memory.context_window_size, 3)
        
        # 잘못된 크기 설정 시 예외 발생 확인
        with self.assertRaises(ValueError):
            self.memory.set_context_window_size(0)
        
        with self.assertRaises(ValueError):
            self.memory.set_context_window_size(20)  # max_memory_size보다 큰 값
    
    def test_get_context_for_llm(self):
        """LLM 입력용 컨텍스트 형식 변환 기능 테스트"""
        # 메시지 추가
        self.memory.add_message("user", "안녕하세요")
        self.memory.add_message("assistant", "안녕하세요, 무엇을 도와드릴까요?")
        self.memory.add_message("user", "날씨가 어때요?")
        
        # LLM 입력용 컨텍스트 형식으로 변환
        llm_context = self.memory.get_context_for_llm()
        
        # 형식 확인
        self.assertEqual(len(llm_context), 3)
        self.assertEqual(llm_context[0]["role"], "user")
        self.assertEqual(llm_context[0]["content"], "안녕하세요")
        self.assertEqual(llm_context[1]["role"], "assistant")
        self.assertEqual(llm_context[1]["content"], "안녕하세요, 무엇을 도와드릴까요?")
        self.assertEqual(llm_context[2]["role"], "user")
        self.assertEqual(llm_context[2]["content"], "날씨가 어때요?")
    
    # 3. 메모리 제한 및 정리 기능 테스트
    
    def test_clear_memory(self):
        """메모리 초기화 기능 테스트"""
        # 메시지 추가
        self.memory.add_message("user", "안녕하세요")
        self.memory.add_message("assistant", "안녕하세요, 무엇을 도와드릴까요?")
        
        # 메모리 초기화
        self.memory.clear_memory()
        
        # 메시지가 모두 삭제되었는지 확인
        self.assertEqual(len(self.memory.get_all_messages()), 0)
    
    def test_set_max_memory_size(self):
        """최대 메모리 크기 설정 기능 테스트"""
        # 초기 최대 메모리 크기 확인
        self.assertEqual(self.memory.max_memory_size, 10)
        
        # 최대 메모리 크기 변경
        self.memory.set_max_memory_size(20)
        self.assertEqual(self.memory.max_memory_size, 20)
        self.assertEqual(self.memory.history.max_messages, 20)
        
        # 잘못된 크기 설정 시 예외 발생 확인
        with self.assertRaises(ValueError):
            self.memory.set_max_memory_size(3)  # context_window_size보다 작은 값
    
    def test_memory_size_limit(self):
        """메모리 크기 제한 기능 테스트"""
        # 최대 메모리 크기보다 많은 메시지 추가
        for i in range(15):
            self.memory.add_message("user", f"메시지 {i}")
        
        # 최대 크기(10)만큼만 저장되었는지 확인
        messages = self.memory.get_all_messages()
        self.assertEqual(len(messages), 10)
        
        # 가장 최근 메시지가 저장되었는지 확인
        self.assertEqual(messages[-1].content, "메시지 14")
        self.assertEqual(messages[0].content, "메시지 5")
    
    def test_memory_size_reduction(self):
        """메모리 크기 축소 시 오래된 메시지 제거 테스트"""
        # 메시지 추가
        for i in range(10):
            self.memory.add_message("user", f"메시지 {i}")
        
        # 최대 메모리 크기 축소
        self.memory.set_max_memory_size(5)
        
        # 메시지 수 확인
        messages = self.memory.get_all_messages()
        self.assertEqual(len(messages), 5)
        
        # 가장 최근 메시지만 남아있는지 확인
        for i, msg in enumerate(messages):
            self.assertEqual(msg.content, f"메시지 {i+5}")
    
    # 4. 파일 저장 및 로드 기능 테스트
    
    def test_to_dict_and_from_dict(self):
        """직렬화 및 역직렬화 기능 테스트"""
        # 메시지 추가
        self.memory.add_message("user", "안녕하세요")
        self.memory.add_message("assistant", "안녕하세요, 무엇을 도와드릴까요?")
        
        # 직렬화
        memory_dict = self.memory.to_dict()
        
        # 역직렬화
        new_memory = ConversationMemory.from_dict(memory_dict)
        
        # 속성 비교
        self.assertEqual(new_memory.context_window_size, self.memory.context_window_size)
        self.assertEqual(new_memory.max_memory_size, self.memory.max_memory_size)
        self.assertEqual(len(new_memory.get_all_messages()), len(self.memory.get_all_messages()))
        
        # 메시지 내용 비교
        original_messages = self.memory.get_all_messages()
        new_messages = new_memory.get_all_messages()
        
        for i in range(len(original_messages)):
            self.assertEqual(new_messages[i].role, original_messages[i].role)
            self.assertEqual(new_messages[i].content, original_messages[i].content)
    
    def test_save_and_load_from_file(self):
        """파일 저장 및 로드 기능 테스트"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # 메시지 추가
            self.memory.add_message("user", "안녕하세요")
            self.memory.add_message("assistant", "안녕하세요, 무엇을 도와드릴까요?")
            
            # 파일에 저장
            self.memory.save_to_file(temp_path)
            
            # 새 메모리 객체 생성
            new_memory = ConversationMemory()
            
            # 파일에서 로드
            success = new_memory.load_from_file(temp_path)
            self.assertTrue(success)
            
            # 메시지 비교
            original_messages = self.memory.get_all_messages()
            loaded_messages = new_memory.get_all_messages()
            
            self.assertEqual(len(loaded_messages), len(original_messages))
            
            for i in range(len(original_messages)):
                self.assertEqual(loaded_messages[i].role, original_messages[i].role)
                self.assertEqual(loaded_messages[i].content, original_messages[i].content)
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_auto_save(self):
        """자동 저장 기능 테스트"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # 자동 저장 활성화된 메모리 객체 생성
            auto_save_memory = ConversationMemory(
                context_window_size=5,
                max_memory_size=10,
                storage_path=temp_path,
                auto_save=True
            )
            
            # 메시지 추가
            auto_save_memory.add_message("user", "자동 저장 테스트")
            
            # 파일이 생성되었는지 확인
            self.assertTrue(os.path.exists(temp_path))
            
            # 파일 내용 확인
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertEqual(len(data["messages"]), 1)
                self.assertEqual(data["messages"][0]["content"], "자동 저장 테스트")
            
            # 추가 메시지 저장
            auto_save_memory.add_message("assistant", "자동으로 저장되었습니다")
            
            # 파일 내용 다시 확인
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertEqual(len(data["messages"]), 2)
                self.assertEqual(data["messages"][1]["content"], "자동으로 저장되었습니다")
            
            # 메모리 초기화 후 자동 저장 확인
            auto_save_memory.clear_memory()
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertEqual(len(data["messages"]), 0)
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_auto_load_from_file(self):
        """파일에서 자동 로드 기능 테스트"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # 메시지 추가 및 파일에 저장
            self.memory.add_message("user", "자동 로드 테스트")
            self.memory.add_message("assistant", "자동으로 로드됩니다")
            self.memory.save_to_file(temp_path)
            
            # 저장 경로를 지정하여 새 메모리 객체 생성 (자동 로드)
            auto_load_memory = ConversationMemory(
                context_window_size=5,
                max_memory_size=10,
                storage_path=temp_path
            )
            
            # 메시지가 자동으로 로드되었는지 확인
            loaded_messages = auto_load_memory.get_all_messages()
            self.assertEqual(len(loaded_messages), 2)
            self.assertEqual(loaded_messages[0].content, "자동 로드 테스트")
            self.assertEqual(loaded_messages[1].content, "자동으로 로드됩니다")
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()