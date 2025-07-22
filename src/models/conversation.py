# src/models/conversation.py

"""
대화 관련 데이터 모델 정의
ConversationMessage 클래스 구현
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class ConversationMessage(BaseModel):
    """
    대화 메시지를 나타내는 모델
    
    Attributes:
        role: 메시지 작성자의 역할 (user, assistant, system)
        content: 메시지 내용
        timestamp: 메시지 생성 시간
        metadata: 추가 메타데이터 (선택 사항)
    """
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('role')
    def validate_role(cls, v):
        """역할이 유효한지 검증"""
        valid_roles = ["user", "assistant", "system"]
        if v not in valid_roles:
            raise ValueError(f'역할은 {", ".join(valid_roles)} 중 하나여야 합니다')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """내용이 비어있지 않은지 검증"""
        if not v or v.strip() == "":
            raise ValueError('내용은 비어있지 않아야 합니다')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        모델을 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 모델의 모든 필드를 포함하는 딕셔너리
        """
        result = self.dict()
        # datetime 객체를 ISO 형식 문자열로 변환
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """
        딕셔너리에서 모델 생성
        
        Args:
            data (Dict[str, Any]): 모델 필드를 포함하는 딕셔너리
            
        Returns:
            ConversationMessage: 생성된 ConversationMessage 객체
        """
        # timestamp가 문자열인 경우 datetime으로 변환
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ConversationHistory(BaseModel):
    """
    대화 기록을 관리하는 모델
    
    Attributes:
        messages: 대화 메시지 목록
        max_messages: 저장할 최대 메시지 수
    """
    messages: List[ConversationMessage] = Field(default_factory=list)
    max_messages: int = 100
    
    def add_message(self, message: ConversationMessage) -> None:
        """
        대화 기록에 메시지 추가
        
        Args:
            message (ConversationMessage): 추가할 메시지
        """
        self.messages.append(message)
        # 최대 메시지 수를 초과하면 가장 오래된 메시지 제거
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, count: Optional[int] = None) -> List[ConversationMessage]:
        """
        최근 메시지 가져오기
        
        Args:
            count (Optional[int], optional): 가져올 메시지 수. 기본값은 None (모든 메시지).
            
        Returns:
            List[ConversationMessage]: 최근 메시지 목록
        """
        if count is None:
            return self.messages
        return self.messages[-count:]
    
    def get_messages_by_role(self, role: str) -> List[ConversationMessage]:
        """
        특정 역할의 메시지만 가져오기
        
        Args:
            role (str): 가져올 메시지의 역할
            
        Returns:
            List[ConversationMessage]: 지정된 역할의 메시지 목록
        """
        return [msg for msg in self.messages if msg.role == role]
    
    def clear(self) -> None:
        """대화 기록 초기화"""
        self.messages = []
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        대화 기록을 딕셔너리 목록으로 변환
        
        Returns:
            List[Dict[str, Any]]: 메시지 딕셔너리 목록
        """
        return [msg.to_dict() for msg in self.messages]
    
    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> 'ConversationHistory':
        """
        딕셔너리 목록에서 대화 기록 생성
        
        Args:
            data (List[Dict[str, Any]]): 메시지 딕셔너리 목록
            
        Returns:
            ConversationHistory: 생성된 ConversationHistory 객체
        """
        history = cls()
        for msg_data in data:
            history.add_message(ConversationMessage.from_dict(msg_data))
        return history