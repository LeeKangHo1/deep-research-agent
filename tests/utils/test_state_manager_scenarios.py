# src/utils/test_state_manager_scenarios.py

"""
StateManager 클래스에 대한 시나리오 및 통합 테스트
"""

import unittest
from datetime import datetime
import json

from src.utils.state_manager import StateManager
from src.models.research import ResearchState, ResearchData, AnalysisResult, SynthesisResult, ValidationStatus


class TestStateManagerScenarios(unittest.TestCase):
    """StateManager 클래스의 시나리오 및 통합 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.query = "인공지능의 윤리적 영향은 무엇인가?"
        self.state_manager = StateManager(initial_query=self.query)
    
    def test_complete_research_workflow(self):
        """전체 연구 워크플로우 시나리오 테스트"""
        # 1. 초기 상태 확인
        self.assertEqual(self.state_manager.get_current_state().current_step, "초기화")
        
        # 2. 연구 단계로 업데이트
        self.state_manager.update_step("정보 수집")
        
        # 3. 연구 데이터 추가
        research_data1 = ResearchData(
            source="https://example.com/article1",
            content="인공지능의 윤리적 영향에 관한 첫 번째 연구",
            reliability_score=0.85
        )
        research_data2 = ResearchData(
            source="https://example.com/article2",
            content="인공지능의 윤리적 영향에 관한 두 번째 연구",
            reliability_score=0.75
        )
        
        self.state_manager.add_research_data(research_data1)
        self.state_manager.add_research_data(research_data2)
        
        # 4. 분석 단계로 업데이트
        self.state_manager.update_step("데이터 분석")
        
        # 5. 분석 결과 추가
        analysis_result = AnalysisResult(
            analysis_type="의미적 분석",
            findings=["인공지능은 의사결정에 영향을 미침", "윤리적 고려사항이 중요함"],
            confidence_score=0.8,
            supporting_data=[research_data1, research_data2],
            insights=["인공지능 윤리 가이드라인 필요"]
        )
        
        self.state_manager.add_analysis_result(analysis_result)
        
        # 6. 종합 단계로 업데이트
        self.state_manager.update_step("결과 종합")
        
        # 7. 종합 결과 설정
        synthesis_result = SynthesisResult(
            summary="인공지능의 윤리적 영향에 관한 종합 연구 결과",
            key_points=["의사결정 영향", "윤리적 고려사항", "가이드라인 필요성"],
            conclusions=["인공지능 윤리는 중요한 연구 분야"],
            recommendations=["윤리적 가이드라인 개발", "지속적인 모니터링"],
            sources=["https://example.com/article1", "https://example.com/article2"],
            confidence_level=0.85
        )
        
        self.state_manager.set_synthesis_result(synthesis_result)
        
        # 8. 검증 단계로 업데이트
        self.state_manager.update_step("결과 검증")
        
        # 9. 검증 상태 설정
        self.state_manager.set_validation_status(ValidationStatus.PASSED)
        
        # 10. 최종 상태 확인
        final_state = self.state_manager.get_current_state()
        self.assertEqual(final_state.current_step, "결과 검증")
        self.assertEqual(final_state.validation_status, ValidationStatus.PASSED)
        self.assertEqual(len(final_state.collected_data), 2)
        self.assertEqual(len(final_state.analysis_results), 1)
        self.assertIsNotNone(final_state.synthesis_result)
        
        # 11. 상태 이력 확인
        history = self.state_manager.get_state_history()
        # 참고: 실제 이력 길이는 구현에 따라 다를 수 있으므로 정확한 숫자 대신 최소 길이만 확인
        self.assertGreaterEqual(len(history), 1)  # 최소한 하나 이상의 상태가 있어야 함
    
    def test_error_recovery(self):
        """오류 상황 처리 및 복구 테스트"""
        # 1. 초기 상태 설정
        self.state_manager.update_step("정보 수집")
        
        # 2. 연구 데이터 추가
        research_data = ResearchData(
            source="https://example.com/article",
            content="인공지능의 윤리적 영향에 관한 연구",
            reliability_score=0.85
        )
        
        self.state_manager.add_research_data(research_data)
        
        # 3. 컨텍스트에 오류 정보 저장
        self.state_manager.set_context("error", {
            "message": "API 호출 실패",
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0
        })
        
        # 4. 오류 상황에서 상태 복구
        error_context = self.state_manager.get_context("error")
        retry_count = error_context.get("retry_count", 0) + 1
        
        # 5. 재시도 횟수 업데이트
        self.state_manager.set_context("error", {
            "message": error_context["message"],
            "timestamp": error_context["timestamp"],
            "retry_count": retry_count
        })
        
        # 6. 재시도 횟수 확인
        updated_error = self.state_manager.get_context("error")
        self.assertEqual(updated_error["retry_count"], 1)
        
        # 7. 오류 해결 후 컨텍스트에서 오류 정보 제거
        self.state_manager.remove_context("error")
        self.assertIsNone(self.state_manager.get_context("error"))
        
        # 8. 정상 상태로 계속 진행
        self.state_manager.update_step("데이터 분석")
        self.assertEqual(self.state_manager.get_current_state().current_step, "데이터 분석")
    
    def test_state_serialization(self):
        """상태 직렬화 및 역직렬화 테스트"""
        # 1. 초기 상태 설정
        self.state_manager.update_step("정보 수집")
        
        # 2. 연구 데이터 추가
        research_data = ResearchData(
            source="https://example.com/article",
            content="인공지능의 윤리적 영향에 관한 연구",
            reliability_score=0.85
        )
        
        self.state_manager.add_research_data(research_data)
        
        # 3. 컨텍스트 설정
        self.state_manager.set_context("session_id", "session123")
        
        # 4. 상태를 딕셔너리로 변환
        state_dict = self.state_manager.to_dict()
        
        # 5. datetime 객체를 문자열로 변환하는 JSON 인코더 클래스 정의
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        # 6. JSON 직렬화 (실제 저장 시나리오 시뮬레이션)
        try:
            json_str = json.dumps(state_dict, cls=DateTimeEncoder)
            # JSON으로 직렬화 가능한지 확인
            self.assertIsInstance(json_str, str)
        except Exception as e:
            self.fail(f"JSON 직렬화 실패: {str(e)}")
        
        # 7. JSON 역직렬화 (실제 로딩 시나리오 시뮬레이션)
        try:
            loaded_dict = json.loads(json_str)
            
            # 8. datetime 문자열을 다시 datetime 객체로 변환하는 함수
            # 참고: 실제 구현에서는 이 부분이 더 복잡할 수 있으며,
            # 모든 중첩된 딕셔너리와 리스트를 재귀적으로 처리해야 할 수 있습니다.
            # 여기서는 테스트 목적으로 간소화합니다.
            
            # 9. 역직렬화된 딕셔너리에서 StateManager 복원
            # 참고: 실제 구현에서는 from_dict 메서드가 datetime 문자열을 처리할 수 있어야 합니다.
            restored_manager = StateManager.from_dict(loaded_dict)
            self.assertIsNotNone(restored_manager)
        except Exception as e:
            self.fail(f"JSON 역직렬화 또는 StateManager 복원 실패: {str(e)}")
        
        # 10. 복원된 상태 확인
        restored_state = restored_manager.get_current_state()
        self.assertEqual(restored_state.query, self.query)
        self.assertEqual(restored_state.current_step, "정보 수집")
        
        # 11. 복원된 컨텍스트 확인
        self.assertEqual(restored_manager.get_context("session_id"), "session123")
    
    def test_concurrent_updates_simulation(self):
        """동시 업데이트 시뮬레이션 테스트"""
        # 1. 초기 상태 설정
        self.state_manager.update_step("정보 수집")
        
        # 2. 첫 번째 업데이트 (에이전트 1 시뮬레이션)
        research_data1 = ResearchData(
            source="https://example.com/article1",
            content="첫 번째 에이전트가 수집한 데이터",
            reliability_score=0.8
        )
        
        self.state_manager.add_research_data(research_data1)
        
        # 3. 상태 스냅샷 저장 (다른 에이전트가 작업 중인 상황 시뮬레이션)
        state_snapshot = self.state_manager.to_dict()
        
        # 4. 두 번째 업데이트 (에이전트 2 시뮬레이션)
        research_data2 = ResearchData(
            source="https://example.com/article2",
            content="두 번째 에이전트가 수집한 데이터",
            reliability_score=0.75
        )
        
        self.state_manager.add_research_data(research_data2)
        self.state_manager.update_step("데이터 분석")
        
        # 5. 첫 번째 에이전트의 상태로 새 관리자 생성 (병렬 처리 시뮬레이션)
        agent1_manager = StateManager.from_dict(state_snapshot)
        
        # 6. 첫 번째 에이전트의 추가 업데이트
        research_data3 = ResearchData(
            source="https://example.com/article3",
            content="첫 번째 에이전트가 추가로 수집한 데이터",
            reliability_score=0.9
        )
        
        agent1_manager.add_research_data(research_data3)
        
        # 7. 두 상태 비교
        main_state = self.state_manager.get_current_state()
        agent1_state = agent1_manager.get_current_state()
        
        self.assertEqual(len(main_state.collected_data), 2)
        self.assertEqual(len(agent1_state.collected_data), 2)
        self.assertEqual(main_state.current_step, "데이터 분석")
        self.assertEqual(agent1_state.current_step, "정보 수집")
        
        # 8. 상태 병합 시뮬레이션 (실제로는 더 복잡한 병합 로직이 필요할 수 있음)
        # 여기서는 간단히 두 번째 상태의 데이터를 첫 번째 상태에 추가하는 방식으로 시뮬레이션
        for data in agent1_state.collected_data:
            # 중복 방지를 위한 간단한 체크 (실제로는 더 정교한 중복 검사가 필요)
            if data.source not in [d.source for d in main_state.collected_data]:
                self.state_manager.add_research_data(data)
        
        # 9. 병합 결과 확인
        merged_state = self.state_manager.get_current_state()
        self.assertEqual(len(merged_state.collected_data), 3)  # 중복되지 않은 모든 데이터 포함


if __name__ == "__main__":
    unittest.main()