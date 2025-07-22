# src/models/test_research_models.py

"""
연구 관련 데이터 모델 테스트 스크립트
"""

from datetime import datetime
from src.models.research import (
    ResearchState,
    ResearchData,
    AnalysisResult,
    SynthesisResult,
    ValidationStatus
)


def test_research_data():
    """ResearchData 모델 테스트"""
    # 기본 객체 생성
    data = ResearchData(
        source="https://example.com/article",
        content="이것은 테스트 콘텐츠입니다.",
        reliability_score=0.85
    )
    
    # 필드 검증
    assert data.source == "https://example.com/article"
    assert data.content == "이것은 테스트 콘텐츠입니다."
    assert data.reliability_score == 0.85
    assert isinstance(data.timestamp, datetime)
    assert isinstance(data.tags, list)
    assert isinstance(data.raw_data, dict)
    
    # 직렬화 및 역직렬화 테스트
    data_dict = data.to_dict()
    restored_data = ResearchData.from_dict(data_dict)
    assert restored_data.source == data.source
    assert restored_data.content == data.content
    assert restored_data.reliability_score == data.reliability_score
    
    print("ResearchData 테스트 통과")


def test_analysis_result():
    """AnalysisResult 모델 테스트"""
    # 테스트 데이터 생성
    research_data = ResearchData(
        source="https://example.com/data",
        content="분석을 위한 테스트 데이터",
        reliability_score=0.9
    )
    
    # 기본 객체 생성
    analysis = AnalysisResult(
        analysis_type="semantic",
        findings=["발견 1", "발견 2"],
        confidence_score=0.75,
        supporting_data=[research_data],
        insights=["인사이트 1"]
    )
    
    # 필드 검증
    assert analysis.analysis_type == "semantic"
    assert len(analysis.findings) == 2
    assert analysis.confidence_score == 0.75
    assert len(analysis.supporting_data) == 1
    assert analysis.supporting_data[0].source == research_data.source
    
    # 직렬화 및 역직렬화 테스트
    analysis_dict = analysis.to_dict()
    restored_analysis = AnalysisResult.from_dict(analysis_dict)
    assert restored_analysis.analysis_type == analysis.analysis_type
    assert restored_analysis.findings == analysis.findings
    assert restored_analysis.confidence_score == analysis.confidence_score
    assert restored_analysis.supporting_data[0].source == analysis.supporting_data[0].source
    
    print("AnalysisResult 테스트 통과")


def test_synthesis_result():
    """SynthesisResult 모델 테스트"""
    # 기본 객체 생성
    synthesis = SynthesisResult(
        summary="이것은 연구 요약입니다.",
        key_points=["핵심 포인트 1", "핵심 포인트 2"],
        conclusions=["결론 1"],
        recommendations=["권장사항 1", "권장사항 2"],
        sources=["https://example.com/source1", "https://example.com/source2"],
        confidence_level=0.8
    )
    
    # 필드 검증
    assert synthesis.summary == "이것은 연구 요약입니다."
    assert len(synthesis.key_points) == 2
    assert len(synthesis.conclusions) == 1
    assert len(synthesis.recommendations) == 2
    assert len(synthesis.sources) == 2
    assert synthesis.confidence_level == 0.8
    
    # 직렬화 및 역직렬화 테스트
    synthesis_dict = synthesis.to_dict()
    restored_synthesis = SynthesisResult.from_dict(synthesis_dict)
    assert restored_synthesis.summary == synthesis.summary
    assert restored_synthesis.key_points == synthesis.key_points
    assert restored_synthesis.conclusions == synthesis.conclusions
    assert restored_synthesis.sources == synthesis.sources
    
    print("SynthesisResult 테스트 통과")


def test_research_state():
    """ResearchState 모델 테스트"""
    # 테스트 데이터 생성
    research_data = ResearchData(
        source="https://example.com/data",
        content="연구 데이터",
        reliability_score=0.85
    )
    
    analysis = AnalysisResult(
        analysis_type="statistical",
        findings=["발견 1"],
        confidence_score=0.7
    )
    
    synthesis = SynthesisResult(
        summary="연구 요약",
        key_points=["핵심 1"],
        conclusions=["결론 1"],
        sources=["https://example.com/source"],
        confidence_level=0.75
    )
    
    # 기본 객체 생성
    state = ResearchState(
        query="인공지능의 미래는?",
        current_step="data_collection"
    )
    
    # 상태 업데이트 테스트
    state.add_research_data(research_data)
    state.add_analysis_result(analysis)
    state.set_synthesis_result(synthesis)
    state.set_validation_status(ValidationStatus.PASSED)
    state.update_step("completed")
    
    # 필드 검증
    assert state.query == "인공지능의 미래는?"
    assert state.current_step == "completed"
    assert len(state.collected_data) == 1
    assert len(state.analysis_results) == 1
    assert state.synthesis_result is not None
    assert state.validation_status == ValidationStatus.PASSED
    
    # 직렬화 및 역직렬화 테스트
    state_dict = state.to_dict()
    restored_state = ResearchState.from_dict(state_dict)
    assert restored_state.query == state.query
    assert restored_state.current_step == state.current_step
    assert len(restored_state.collected_data) == len(state.collected_data)
    assert restored_state.validation_status == state.validation_status
    
    print("ResearchState 테스트 통과")


if __name__ == "__main__":
    print("연구 관련 데이터 모델 테스트 시작...")
    test_research_data()
    test_analysis_result()
    test_synthesis_result()
    test_research_state()
    print("모든 테스트 통과!")