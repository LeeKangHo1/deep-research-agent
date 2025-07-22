# src/models/research.py

"""
연구 관련 데이터 모델 정의
ResearchState, ResearchData, AnalysisResult, SynthesisResult 클래스 구현
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class ValidationStatus(str, Enum):
    """검증 상태를 나타내는 열거형"""
    NOT_STARTED = "not_started"  # 검증이 시작되지 않음
    IN_PROGRESS = "in_progress"  # 검증이 진행 중
    PASSED = "passed"            # 검증 통과
    FAILED = "failed"            # 검증 실패
    PARTIAL = "partial"          # 일부 검증 통과


class ResearchData(BaseModel):
    """
    연구 과정에서 수집된 데이터를 나타내는 모델
    
    Attributes:
        source: 데이터 출처 (URL, 문서명 등)
        content: 데이터 내용
        reliability_score: 신뢰도 점수 (0.0 ~ 1.0)
        timestamp: 데이터 수집 시간
        tags: 데이터 관련 태그 목록
        raw_data: 원본 데이터 (JSON 형태)
    """
    source: str
    content: str
    reliability_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('reliability_score')
    def validate_reliability_score(cls, v):
        """신뢰도 점수가 0.0에서 1.0 사이인지 검증"""
        if v < 0.0 or v > 1.0:
            raise ValueError('신뢰도 점수는 0.0에서 1.0 사이여야 합니다')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        모델을 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 모델의 모든 필드를 포함하는 딕셔너리
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchData':
        """
        딕셔너리에서 모델 생성
        
        Args:
            data (Dict[str, Any]): 모델 필드를 포함하는 딕셔너리
            
        Returns:
            ResearchData: 생성된 ResearchData 객체
        """
        return cls(**data)


class AnalysisResult(BaseModel):
    """
    데이터 분석 결과를 나타내는 모델
    
    Attributes:
        analysis_type: 분석 유형 (통계적, 의미적, 비교 등)
        findings: 발견된 사실이나 결과 목록
        confidence_score: 분석 결과의 신뢰도 (0.0 ~ 1.0)
        supporting_data: 분석을 뒷받침하는 데이터 목록
        insights: 분석에서 도출된 인사이트 목록
    """
    analysis_type: str
    findings: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_data: List[ResearchData] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """신뢰도 점수가 0.0에서 1.0 사이인지 검증"""
        if v < 0.0 or v > 1.0:
            raise ValueError('신뢰도 점수는 0.0에서 1.0 사이여야 합니다')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return {
            "analysis_type": self.analysis_type,
            "findings": self.findings,
            "confidence_score": self.confidence_score,
            "supporting_data": [data.to_dict() for data in self.supporting_data],
            "insights": self.insights
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """딕셔너리에서 모델 생성"""
        if "supporting_data" in data:
            data["supporting_data"] = [
                ResearchData.from_dict(item) for item in data["supporting_data"]
            ]
        return cls(**data)


class SynthesisResult(BaseModel):
    """
    분석 결과를 종합한 최종 결과를 나타내는 모델
    
    Attributes:
        summary: 전체 연구 결과 요약
        key_points: 주요 포인트 목록
        conclusions: 결론 목록
        recommendations: 권장사항 목록
        sources: 참고 출처 목록
        confidence_level: 종합 결과의 신뢰도 (0.0 ~ 1.0)
    """
    summary: str
    key_points: List[str]
    conclusions: List[str]
    recommendations: List[str] = Field(default_factory=list)
    sources: List[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        """신뢰도 수준이 0.0에서 1.0 사이인지 검증"""
        if v < 0.0 or v > 1.0:
            raise ValueError('신뢰도 수준은 0.0에서 1.0 사이여야 합니다')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynthesisResult':
        """딕셔너리에서 모델 생성"""
        return cls(**data)


class ResearchState(BaseModel):
    """
    연구 프로세스의 현재 상태를 나타내는 모델
    
    Attributes:
        query: 사용자의 원래 질문
        current_step: 현재 진행 중인 연구 단계
        collected_data: 수집된 연구 데이터 목록
        analysis_results: 분석 결과 목록
        synthesis_result: 종합 결과 (있는 경우)
        validation_status: 검증 상태
        metadata: 추가 메타데이터
    """
    query: str
    current_step: str
    collected_data: List[ResearchData] = Field(default_factory=list)
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    synthesis_result: Optional[SynthesisResult] = None
    validation_status: ValidationStatus = ValidationStatus.NOT_STARTED
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        result = {
            "query": self.query,
            "current_step": self.current_step,
            "collected_data": [data.to_dict() for data in self.collected_data],
            "analysis_results": [result.to_dict() for result in self.analysis_results],
            "validation_status": self.validation_status,
            "metadata": self.metadata
        }
        
        if self.synthesis_result:
            result["synthesis_result"] = self.synthesis_result.to_dict()
        else:
            result["synthesis_result"] = None
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchState':
        """딕셔너리에서 모델 생성"""
        if "collected_data" in data:
            data["collected_data"] = [
                ResearchData.from_dict(item) for item in data["collected_data"]
            ]
        
        if "analysis_results" in data:
            data["analysis_results"] = [
                AnalysisResult.from_dict(item) for item in data["analysis_results"]
            ]
            
        if "synthesis_result" in data and data["synthesis_result"]:
            data["synthesis_result"] = SynthesisResult.from_dict(data["synthesis_result"])
            
        return cls(**data)
    
    def update_step(self, step: str) -> None:
        """
        현재 단계 업데이트
        
        Args:
            step (str): 새로운 연구 단계 이름
        """
        self.current_step = step
        
    def add_research_data(self, data: ResearchData) -> None:
        """
        연구 데이터 추가
        
        Args:
            data (ResearchData): 추가할 연구 데이터 객체
        """
        self.collected_data.append(data)
        
    def add_analysis_result(self, result: AnalysisResult) -> None:
        """
        분석 결과 추가
        
        Args:
            result (AnalysisResult): 추가할 분석 결과 객체
        """
        self.analysis_results.append(result)
        
    def set_synthesis_result(self, result: SynthesisResult) -> None:
        """
        종합 결과 설정
        
        Args:
            result (SynthesisResult): 설정할 종합 결과 객체
        """
        self.synthesis_result = result
        
    def set_validation_status(self, status: ValidationStatus) -> None:
        """
        검증 상태 설정
        
        Args:
            status (ValidationStatus): 설정할 검증 상태
        """
        self.validation_status = status