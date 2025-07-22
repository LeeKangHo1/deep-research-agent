# src/models/test_conversation_models.py

"""
대화 관련 데이터 모델 테스트
"""

import unittest
from datetime import datetime
from .conversation import ConversationMessage


class TestConversationMessage(unittest.TestCase):
    """ConversationMessage 클래스 테스트"""
    
    def test_create_conversation_message(self):
        """기본 ConversationMessage 객체 생성 테스트"""
        message = ConversationMessage(
            role="user",
            content="안녕하세요, 질문이 있습니다."
        )
        
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "안녕하세요, 질문이 있습니다.")
        self.assertIsNotNone(message.timestamp)
        self.assertIsNone(message.metadata)
        
    def test_create_conversation_message_with_metadata(self):
        """메타데이터를 포함한 ConversationMessage 객체 생성 테스트"""
        now = datetime.now()
        metadata = {"client_id": "12345", "session_id": "abcde"}
        
        message = ConversationMessage(
            role="assistant",
            content="안녕하세요! 어떻게 도와드릴까요?",
            timestamp=now,
            metadata=metadata
        )
        
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "안녕하세요! 어떻게 도와드릴까요?")
        self.assertEqual(message.timestamp, now)
        self.assertEqual(message.metadata, metadata)
        
    def test_invalid_role(self):
        """유효하지 않은 역할로 ConversationMessage 객체 생성 시도"""
        with self.assertRaises(ValueError):
            ConversationMessage(
                role="invalid_role",  # 유효하지 않은 역할
                content="테스트 메시지"
            )
            
    def test_empty_content(self):
        """빈 내용으로 ConversationMessage 객체 생성 시도"""
        with self.assertRaises(ValueError):
            ConversationMessage(
                role="user",
                content=""  # 빈 문자열
            )
            
        with self.assertRaises(ValueError):
            ConversationMessage(
                role="user",
                content="   "  # 공백만 있는 문자열
            )
            
    def test_invalid_content_type(self):
        """유효하지 않은 내용 타입으로 ConversationMessage 객체 생성 시도"""
        with self.assertRaises(Exception):
            ConversationMessage(
                role="user",
                content=123  # 문자열이 아님
            )
            
    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        now = datetime.now()
        metadata = {"client_id": "12345"}
        
        message = ConversationMessage(
            role="system",
            content="시스템 메시지입니다.",
            timestamp=now,
            metadata=metadata
        )
        
        message_dict = message.to_dict()
        self.assertEqual(message_dict["role"], "system")
        self.assertEqual(message_dict["content"], "시스템 메시지입니다.")
        self.assertEqual(message_dict["timestamp"], now.isoformat())
        self.assertEqual(message_dict["metadata"], metadata)
        
    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        now = datetime.now()
        data = {
            "role": "user",
            "content": "사용자 메시지입니다.",
            "timestamp": now.isoformat(),
            "metadata": {"source": "web"}
        }
        
        message = ConversationMessage.from_dict(data)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "사용자 메시지입니다.")
        self.assertEqual(message.timestamp.isoformat(), now.isoformat())
        self.assertEqual(message.metadata, {"source": "web"})
        
    def test_from_dict_without_timestamp(self):
        """timestamp가 없는 딕셔너리에서 ConversationMessage 객체 생성 테스트"""
        data = {
            "role": "user",
            "content": "사용자 메시지입니다."
        }
        
        message = ConversationMessage.from_dict(data)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "사용자 메시지입니다.")
        self.assertIsNotNone(message.timestamp)
        self.assertIsNone(message.metadata)


if __name__ == "__main__":
    unittest.main()