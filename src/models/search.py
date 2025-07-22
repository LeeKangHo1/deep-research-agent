# src/models/search.py

"""
검색 관련 데이터 모델 정의
SearchResult 클래스 구현
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl


class SearchResult(BaseModel):
    """
    검색 결과를 나타내는 모델
    
    Attributes:
        title: 검색 결과 제목
        url: 검색 결과 URL
        content: 검색 결과 내용
        score: 검색 결과 관련성 점수 (0.0 ~ 1.0)
        published_date: 콘텐츠 발행 날짜 (선택 사항)
        snippet: 검색 결과 스니펫 (선택 사항)
        source_type: 검색 결과 소스 유형 (웹페이지, PDF 등) (선택 사항)
        metadata: 추가 메타데이터 (선택 사항)
    """
    title: str
    url: HttpUrl
    content: str
    score: float = Field(ge=0.0, le=1.0)
    published_date: Optional[datetime] = None
    snippet: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('score')
    def validate_score(cls, v):
        """점수가 0.0에서 1.0 사이인지 검증"""
        if v < 0.0 or v > 1.0:
            raise ValueError('점수는 0.0에서 1.0 사이여야 합니다')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        모델을 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 모델의 모든 필드를 포함하는 딕셔너리
        """
        result = self.dict()
        # datetime 객체를 ISO 형식 문자열로 변환
        if self.published_date:
            result["published_date"] = self.published_date.isoformat()
        # HttpUrl 객체를 문자열로 변환
        result["url"] = str(self.url)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """
        딕셔너리에서 모델 생성
        
        Args:
            data (Dict[str, Any]): 모델 필드를 포함하는 딕셔너리
            
        Returns:
            SearchResult: 생성된 SearchResult 객체
        """
        # published_date가 문자열인 경우 datetime으로 변환
        if "published_date" in data and isinstance(data["published_date"], str):
            data["published_date"] = datetime.fromisoformat(data["published_date"])
        return cls(**data)
    
    def get_summary(self, max_length: int = 150) -> str:
        """
        검색 결과의 요약 반환
        
        Args:
            max_length (int, optional): 최대 요약 길이. 기본값은 150.
            
        Returns:
            str: 스니펫이 있으면 스니펫, 없으면 내용의 요약
        """
        # 스니펫이 있으면 스니펫 반환
        if self.snippet:
            summary = self.snippet
        else:
            # 스니펫이 없으면 내용 사용
            summary = self.content
            
        # 최대 길이 제한
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    def get_formatted_date(self, format_str: str = "%Y-%m-%d") -> Optional[str]:
        """
        발행 날짜를 지정된 형식으로 반환
        
        Args:
            format_str (str, optional): 날짜 형식 문자열. 기본값은 "%Y-%m-%d".
            
        Returns:
            Optional[str]: 형식화된 날짜 문자열 또는 날짜가 없는 경우 None
        """
        if self.published_date:
            return self.published_date.strftime(format_str)
        return None