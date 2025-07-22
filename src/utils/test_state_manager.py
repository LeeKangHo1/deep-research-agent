# src/utils/test_state_manager.py

"""
StateManager 클래스에 대한 단위 테스트
"""

import unittest
from datetime import datetime

from src.utils.state_manager import StateManager
from src.models.research import ResearchState, ResearchData, ValidationStatus


class TestStateManager(unittest.TestCase):
    """StateManager 클래스 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.query = "인공지능의 윤리적 영향은 무엇인가?"
        self.state_manager = StateManager(initial_query=self.query)
    
    def test_initialization(self):
        """초기화 테스트"""
        # 초기 상태 확인
        current_state = self.state_manager.get_current_state()
        self.assertIsNotNone(current_state)
        self.assertEqual(current_state.query, self.query)
        self.assertEqual(current_state.current_step, "초기화")
        self.assertEqual(current_state.validation_status, ValidationStatus.NOT_STARTED)
        
        # 상태 이력 확인
        history = self.state_manager.get_state_history()
        self.assertEqual(len(history), 1)
        
        # 컨텍스트 확인
        self.assertEqual(self.state_manager.context, {})
    
    def test_initialize_state(self):
        """상태 초기화 테스트"""
        new_query = "기후 변화의 경제적 영향은?"
        state = self.state_manager.initialize_state(new_query)
        
        # 새로운 상태 확인
        self.assertEqual(state.query, new_query)
        self.assertEqual(state.current_step, "초기화")
        
        # 상태 이력 확인
        history = self.state_manager.get_state_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].query, new_query)
    
    def test_get_state_methods(self):
        """상태 조회 메서드 테스트"""
        # 현재 상태 조회
        current_state = self.state_manager.get_current_state()
        self.assertIsNotNone(current_state)
        
        # 마지막 상태 조회
        last_state = self.state_manager.get_last_state()
        self.assertIsNotNone(last_state)
        self.assertEqual(last_state.query, self.query)
        
        # 인덱스로 상태 조회
        state_at_index = self.state_manager.get_state_at_index(0)
        self.assertIsNotNone(state_at_index)
        self.assertEqual(state_at_index.query, self.query)
        
        # 잘못된 인덱스로 상태 조회
        invalid_state = self.state_manager.get_state_at_index(99)
        self.assertIsNone(invalid_state)
    
    def test_to_dict_and_from_dict(self):
        """딕셔너리 변환 및 복원 테스트"""
        # 딕셔너리로 변환
        state_dict = self.state_manager.to_dict()
        self.assertIsNotNone(state_dict)
        self.assertIn("current_state", state_dict)
        self.assertIn("state_history", state_dict)
        self.assertIn("context", state_dict)
        
        # 딕셔너리에서 복원
        restored_manager = StateManager.from_dict(state_dict)
        self.assertIsNotNone(restored_manager)
        
        # 복원된 객체 확인
        restored_state = restored_manager.get_current_state()
        self.assertEqual(restored_state.query, self.query)
        self.assertEqual(restored_state.current_step, "초기화")
    
    def test_update_state(self):
        """상태 업데이트 테스트"""
        # 상태 업데이트
        updates = {"current_step": "정보 수집"}
        updated_state = self.state_manager.update_state(updates)
        
        # 업데이트된 상태 확인
        self.assertEqual(updated_state.current_step, "정보 수집")
        self.assertEqual(updated_state.query, self.query)  # 기존 값은 유지
        
        # 상태 이력 확인
        history = self.state_manager.get_state_history()
        self.assertEqual(len(history), 2)  # 초기 상태 + 업데이트된 상태
        self.assertEqual(history[-1].current_step, "정보 수집")
    
    def test_update_step(self):
        """연구 단계 업데이트 테스트"""
        # 연구 단계 업데이트
        new_step = "데이터 분석"
        updated_state = self.state_manager.update_step(new_step)
        
        # 업데이트된 상태 확인
        self.assertEqual(updated_state.current_step, new_step)
        
        # 상태 이력 확인
        history = self.state_manager.get_state_history()
        self.assertEqual(len(history), 2)  # 초기 상태 + 업데이트된 상태
        self.assertEqual(history[-1].current_step, new_step)
    
    def test_add_research_data(self):
        """연구 데이터 추가 테스트"""
        # 연구 데이터 생성
        research_data = ResearchData(
            source="https://example.com",
            content="인공지능의 윤리적 영향에 관한 연구",
            reliability_score=0.85
        )
        
        # 연구 데이터 추가
        updated_state = self.state_manager.add_research_data(research_data)
        
        # 업데이트된 상태 확인
        self.assertEqual(len(updated_state.collected_data), 1)
        self.assertEqual(updated_state.collected_data[0].source, "https://example.com")
        self.assertEqual(updated_state.collected_data[0].reliability_score, 0.85)
    
    def test_validate_state(self):
        """상태 유효성 검사 테스트"""
        # 유효한 상태 검사
        is_valid = self.state_manager.validate_state()
        self.assertTrue(is_valid)
        
        # 새로운 상태 관리자 생성 (초기화되지 않은 상태)
        empty_manager = StateManager()
        
        # 초기화되지 않은 상태 검사
        is_valid = empty_manager.validate_state()
        self.assertFalse(is_valid)
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        # 초기화되지 않은 상태 관리자 생성
        empty_manager = StateManager()
        
        # 초기화되지 않은 상태에서 메서드 호출 시 예외 발생 확인
        with self.assertRaises(ValueError):
            empty_manager.update_step("데이터 분석")
            
        with self.assertRaises(ValueError):
            empty_manager.add_research_data(ResearchData(
                source="https://example.com",
                content="테스트 데이터",
                reliability_score=0.8
            ))
    
    def test_context_management(self):
        """컨텍스트 관리 기능 테스트"""
        # 컨텍스트 설정
        self.state_manager.set_context("user_id", "user123")
        self.state_manager.set_context("session_start", datetime.now())
        
        # 컨텍스트 조회
        user_id = self.state_manager.get_context("user_id")
        self.assertEqual(user_id, "user123")
        
        # 존재하지 않는 키 조회 (기본값 반환)
        default_value = self.state_manager.get_context("non_existent_key", "default")
        self.assertEqual(default_value, "default")
        
        # 컨텍스트 제거
        removed = self.state_manager.remove_context("user_id")
        self.assertTrue(removed)
        self.assertIsNone(self.state_manager.get_context("user_id"))
        
        # 존재하지 않는 키 제거
        removed = self.state_manager.remove_context("non_existent_key")
        self.assertFalse(removed)
    
    def test_context_update_and_merge(self):
        """컨텍스트 업데이트 및 병합 테스트"""
        # 초기 컨텍스트 설정
        self.state_manager.set_context("key1", "value1")
        self.state_manager.set_context("key2", "value2")
        
        # 컨텍스트 업데이트
        self.state_manager.update_context({
            "key2": "updated_value2",
            "key3": "value3"
        })
        
        # 업데이트 결과 확인
        self.assertEqual(self.state_manager.get_context("key1"), "value1")
        self.assertEqual(self.state_manager.get_context("key2"), "updated_value2")
        self.assertEqual(self.state_manager.get_context("key3"), "value3")
        
        # 전체 컨텍스트 조회
        full_context = self.state_manager.get_full_context()
        self.assertEqual(len(full_context), 3)
        self.assertEqual(full_context["key1"], "value1")
        
        # 컨텍스트 병합 (덮어쓰기)
        self.state_manager.merge_context({
            "key1": "merged_value1",
            "key4": "value4"
        })
        
        # 병합 결과 확인 (덮어쓰기)
        self.assertEqual(self.state_manager.get_context("key1"), "merged_value1")
        self.assertEqual(self.state_manager.get_context("key4"), "value4")
        
        # 컨텍스트 병합 (덮어쓰지 않음)
        self.state_manager.merge_context({
            "key1": "should_not_change",
            "key5": "value5"
        }, overwrite=False)
        
        # 병합 결과 확인 (덮어쓰지 않음)
        self.assertEqual(self.state_manager.get_context("key1"), "merged_value1")  # 변경되지 않음
        self.assertEqual(self.state_manager.get_context("key5"), "value5")  # 새로운 키는 추가됨
    
    def test_context_with_state(self):
        """상태 정보를 포함한 컨텍스트 테스트"""
        # 컨텍스트 설정
        self.state_manager.set_context("session_id", "session123")
        
        # 상태 정보를 포함한 컨텍스트 조회
        context_with_state = self.state_manager.get_context_with_state()
        
        # 컨텍스트 확인
        self.assertEqual(context_with_state["session_id"], "session123")
        
        # 상태 정보 확인
        self.assertIn("current_state", context_with_state)
        self.assertEqual(context_with_state["current_state"]["query"], self.query)
        self.assertEqual(context_with_state["current_state"]["current_step"], "초기화")
        
        # 컨텍스트 초기화
        self.state_manager.clear_context()
        
        # 초기화 확인
        self.assertEqual(len(self.state_manager.get_full_context()), 0)
        
        # 상태 정보는 유지됨
        context_after_clear = self.state_manager.get_context_with_state()
        self.assertIn("current_state", context_after_clear)


if __name__ == "__main__":
    unittest.main()