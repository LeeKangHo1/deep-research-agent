# src/utils/state_manager.py

"""
상태 관리자 클래스 구현
연구 프로세스의 상태를 관리하고 컨텍스트를 유지하는 기능 제공
"""

from typing import Dict, Any, List, Optional
from copy import deepcopy
from datetime import datetime

from src.models.research import ResearchState, ResearchData, AnalysisResult, SynthesisResult, ValidationStatus
from src.models.conversation import ConversationHistory, ConversationMessage


class StateManager:
    """
    연구 프로세스의 상태를 관리하는 클래스
    
    상태 저장, 업데이트, 컨텍스트 관리 등의 기능을 제공합니다.
    
    Attributes:
        current_state (ResearchState): 현재 연구 상태
        state_history (List[ResearchState]): 상태 변경 이력
        context (Dict[str, Any]): 현재 컨텍스트 정보
        max_history_size (int): 저장할 최대 상태 이력 수
    """
    
    def __init__(self, initial_query: Optional[str] = None):
        """
        StateManager 초기화
        
        Args:
            initial_query (Optional[str], optional): 초기 연구 질문. 기본값은 None.
        """
        # 초기 상태 생성
        if initial_query:
            self.current_state = ResearchState(
                query=initial_query,
                current_step="초기화"
            )
        else:
            self.current_state = None
            
        # 상태 이력 저장을 위한 리스트
        self.state_history = []
        
        # 컨텍스트 정보를 저장하기 위한 딕셔너리
        self.context = {}
        
        # 최대 상태 이력 크기
        self.max_history_size = 10
        
        # 초기 상태가 있으면 이력에 추가
        if self.current_state:
            self._add_to_history()
    
    def initialize_state(self, query: str) -> ResearchState:
        """
        새로운 연구 상태 초기화
        
        Args:
            query (str): 연구 질문
            
        Returns:
            ResearchState: 초기화된 연구 상태
        """
        self.current_state = ResearchState(
            query=query,
            current_step="초기화"
        )
        # 상태 이력 초기화
        self.state_history = []
        # 상태 이력에 현재 상태 추가
        self._add_to_history()
        
        return self.current_state
    
    def _add_to_history(self) -> None:
        """
        현재 상태를 이력에 추가
        이력 크기가 최대값을 초과하면 가장 오래된 상태 제거
        """
        if self.current_state:
            # 현재 상태의 깊은 복사본을 이력에 추가
            self.state_history.append(deepcopy(self.current_state))
            
            # 최대 이력 크기를 초과하면 가장 오래된 상태 제거
            if len(self.state_history) > self.max_history_size:
                self.state_history = self.state_history[-self.max_history_size:]
    
    def get_current_state(self) -> Optional[ResearchState]:
        """
        현재 연구 상태 반환
        
        Returns:
            Optional[ResearchState]: 현재 연구 상태 또는 None
        """
        return self.current_state
    
    def get_state_history(self) -> List[ResearchState]:
        """
        상태 변경 이력 반환
        
        Returns:
            List[ResearchState]: 상태 변경 이력 목록
        """
        return self.state_history
    
    def get_state_at_index(self, index: int) -> Optional[ResearchState]:
        """
        특정 인덱스의 상태 반환
        
        Args:
            index (int): 상태 이력 인덱스
            
        Returns:
            Optional[ResearchState]: 지정된 인덱스의 상태 또는 None
        """
        if 0 <= index < len(self.state_history):
            return self.state_history[index]
        return None
    
    def get_last_state(self) -> Optional[ResearchState]:
        """
        마지막으로 저장된 상태 반환
        
        Returns:
            Optional[ResearchState]: 마지막 상태 또는 None
        """
        if self.state_history:
            return self.state_history[-1]
        return None
    
    def update_state(self, updates: Dict[str, Any]) -> ResearchState:
        """
        현재 상태 업데이트
        
        Args:
            updates (Dict[str, Any]): 업데이트할 필드와 값을 포함하는 딕셔너리
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        # 현재 상태를 딕셔너리로 변환
        state_dict = self.current_state.to_dict()
        
        # 업데이트 적용
        for key, value in updates.items():
            if key in state_dict:
                state_dict[key] = value
        
        # 업데이트된 딕셔너리로 새 상태 생성
        self.current_state = ResearchState.from_dict(state_dict)
        
        # 상태 이력에 추가
        self._add_to_history()
        
        return self.current_state
    
    def update_step(self, step: str) -> ResearchState:
        """
        현재 연구 단계 업데이트
        
        Args:
            step (str): 새로운 연구 단계
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        self.current_state.update_step(step)
        self._add_to_history()
        
        return self.current_state
    
    def add_research_data(self, data: ResearchData) -> ResearchState:
        """
        연구 데이터 추가
        
        Args:
            data (ResearchData): 추가할 연구 데이터
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        self.current_state.add_research_data(data)
        self._add_to_history()
        
        return self.current_state
    
    def add_analysis_result(self, result: AnalysisResult) -> ResearchState:
        """
        분석 결과 추가
        
        Args:
            result (AnalysisResult): 추가할 분석 결과
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        self.current_state.add_analysis_result(result)
        self._add_to_history()
        
        return self.current_state
    
    def set_synthesis_result(self, result: SynthesisResult) -> ResearchState:
        """
        종합 결과 설정
        
        Args:
            result (SynthesisResult): 설정할 종합 결과
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        self.current_state.set_synthesis_result(result)
        self._add_to_history()
        
        return self.current_state
    
    def set_validation_status(self, status: ValidationStatus) -> ResearchState:
        """
        검증 상태 설정
        
        Args:
            status (ValidationStatus): 설정할 검증 상태
            
        Returns:
            ResearchState: 업데이트된 연구 상태
            
        Raises:
            ValueError: 현재 상태가 초기화되지 않은 경우
        """
        if not self.current_state:
            raise ValueError("상태가 초기화되지 않았습니다. initialize_state()를 먼저 호출하세요.")
        
        self.current_state.set_validation_status(status)
        self._add_to_history()
        
        return self.current_state
    
    def validate_state(self) -> bool:
        """
        현재 상태의 유효성 검사
        
        Returns:
            bool: 상태가 유효하면 True, 그렇지 않으면 False
        """
        if not self.current_state:
            return False
        
        # 필수 필드 검사
        if not self.current_state.query or not self.current_state.current_step:
            return False
        
        # 추가적인 유효성 검사 로직을 여기에 구현할 수 있습니다.
        # 예: 특정 단계에서 필요한 데이터가 있는지 확인
        
        return True
    
    # 컨텍스트 관리 기능
    
    def set_context(self, key: str, value: Any) -> None:
        """
        컨텍스트에 값 설정
        
        Args:
            key (str): 컨텍스트 키
            value (Any): 저장할 값
        """
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        컨텍스트에서 값 조회
        
        Args:
            key (str): 컨텍스트 키
            default (Any, optional): 키가 없을 경우 반환할 기본값. 기본값은 None.
            
        Returns:
            Any: 컨텍스트 값 또는 기본값
        """
        return self.context.get(key, default)
    
    def remove_context(self, key: str) -> bool:
        """
        컨텍스트에서 값 제거
        
        Args:
            key (str): 제거할 컨텍스트 키
            
        Returns:
            bool: 제거 성공 여부
        """
        if key in self.context:
            del self.context[key]
            return True
        return False
    
    def clear_context(self) -> None:
        """컨텍스트 초기화"""
        self.context = {}
    
    def update_context(self, new_context: Dict[str, Any]) -> None:
        """
        여러 컨텍스트 값 업데이트
        
        Args:
            new_context (Dict[str, Any]): 업데이트할 컨텍스트 키-값 쌍
        """
        self.context.update(new_context)
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        전체 컨텍스트 반환
        
        Returns:
            Dict[str, Any]: 현재 컨텍스트 전체
        """
        return self.context.copy()
    
    def merge_context(self, other_context: Dict[str, Any], overwrite: bool = True) -> None:
        """
        다른 컨텍스트와 병합
        
        Args:
            other_context (Dict[str, Any]): 병합할 컨텍스트
            overwrite (bool, optional): 충돌 시 덮어쓰기 여부. 기본값은 True.
        """
        if overwrite:
            self.context.update(other_context)
        else:
            # 기존 키는 유지하고 새로운 키만 추가
            for key, value in other_context.items():
                if key not in self.context:
                    self.context[key] = value
    
    def get_context_with_state(self) -> Dict[str, Any]:
        """
        현재 상태 정보를 포함한 컨텍스트 반환
        
        Returns:
            Dict[str, Any]: 상태 정보가 포함된 컨텍스트
        """
        result = self.get_full_context()
        
        if self.current_state:
            # 현재 상태 정보를 컨텍스트에 추가
            result["current_state"] = {
                "query": self.current_state.query,
                "current_step": self.current_state.current_step,
                "validation_status": self.current_state.validation_status,
                "collected_data_count": len(self.current_state.collected_data),
                "analysis_results_count": len(self.current_state.analysis_results),
                "has_synthesis_result": self.current_state.synthesis_result is not None
            }
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        상태 관리자 정보를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 상태 관리자 정보를 포함하는 딕셔너리
        """
        result = {
            "context": self.context,
            "max_history_size": self.max_history_size
        }
        
        if self.current_state:
            result["current_state"] = self.current_state.to_dict()
        else:
            result["current_state"] = None
            
        result["state_history"] = [
            state.to_dict() for state in self.state_history
        ] if self.state_history else []
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateManager':
        """
        딕셔너리에서 상태 관리자 생성
        
        Args:
            data (Dict[str, Any]): 상태 관리자 정보를 포함하는 딕셔너리
            
        Returns:
            StateManager: 생성된 StateManager 객체
        """
        manager = cls()
        
        if "current_state" in data and data["current_state"]:
            manager.current_state = ResearchState.from_dict(data["current_state"])
            
        if "state_history" in data and data["state_history"]:
            manager.state_history = [
                ResearchState.from_dict(state_data) for state_data in data["state_history"]
            ]
            
        if "context" in data:
            manager.context = data["context"]
            
        if "max_history_size" in data:
            manager.max_history_size = data["max_history_size"]
            
        return manager