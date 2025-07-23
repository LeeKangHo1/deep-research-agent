# src/models/vector_db.py

"""
벡터 데이터베이스 관련 데이터 모델 정의
VectorDBEntry 클래스 구현
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class VectorDBEntry(BaseModel):
    """
    벡터 데이터베이스에 저장되는 항목을 나타내는 모델
    
    Attributes:
        content: 저장된 텍스트 내용
        metadata: 메타데이터 (검색 쿼리, 타임스탬프, 소스 URL, 검색 엔진 등)
        vector: 임베딩 벡터
        id: 고유 식별자
        timestamp: 데이터 저장 시간
        ttl: 데이터 유효 기간 (초 단위, 선택 사항)
    """
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector: List[float] = Field(default_factory=list)
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl: Optional[int] = None  # 데이터 유효 기간 (초)
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDBEntry':
        """
        딕셔너리에서 모델 생성
        
        Args:
            data (Dict[str, Any]): 모델 필드를 포함하는 딕셔너리
            
        Returns:
            VectorDBEntry: 생성된 VectorDBEntry 객체
        """
        # timestamp가 문자열인 경우 datetime으로 변환
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """
        항목이 만료되었는지 확인
        
        Returns:
            bool: 항목이 만료되었으면 True, 그렇지 않으면 False
        """
        if self.ttl is None:
            return False
        
        current_time = datetime.now()
        expiry_time = self.timestamp.timestamp() + self.ttl
        return current_time.timestamp() > expiry_time
    
    def get_age_in_days(self) -> float:
        """
        항목의 나이를 일 단위로 반환
        
        Returns:
            float: 항목이 저장된 이후 경과한 일수
        """
        current_time = datetime.now()
        age_seconds = (current_time - self.timestamp).total_seconds()
        return age_seconds / (24 * 60 * 60)  # 초를 일로 변환