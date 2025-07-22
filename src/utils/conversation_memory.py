# src/utils/conversation_memory.py

"""
대화 메모리 시스템 구현
대화 기록을 저장하고 관리하는 기능 제공
"""

from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import json
import os
import re
from pathlib import Path

from src.models.conversation import ConversationMessage, ConversationHistory


class ConversationMemory:
    """
    대화 기록을 저장하고 관리하는 클래스
    
    대화 메시지 저장, 조회, 컨텍스트 윈도우 관리 등의 기능을 제공합니다.
    메모리 크기 제한 및 파일 기반 저장 기능을 지원합니다.
    
    Attributes:
        history (ConversationHistory): 대화 기록
        context_window_size (int): 컨텍스트 윈도우 크기 (최근 메시지 수)
        max_memory_size (int): 저장할 최대 메시지 수
        storage_path (Optional[str]): 대화 기록 저장 경로
        auto_save (bool): 자동 저장 활성화 여부
    """
    
    def __init__(self, context_window_size: int = 10, max_memory_size: int = 100, 
                 storage_path: Optional[str] = None, auto_save: bool = False):
        """
        ConversationMemory 초기화
        
        Args:
            context_window_size (int, optional): 컨텍스트 윈도우 크기. 기본값은 10.
            max_memory_size (int, optional): 저장할 최대 메시지 수. 기본값은 100.
            storage_path (Optional[str], optional): 대화 기록 저장 경로. 기본값은 None.
            auto_save (bool, optional): 메시지 추가 시 자동 저장 여부. 기본값은 False.
        """
        self.history = ConversationHistory(max_messages=max_memory_size)
        self.context_window_size = context_window_size
        self.max_memory_size = max_memory_size
        self.storage_path = storage_path
        self.auto_save = auto_save
        
        # 저장 경로가 지정되었고 파일이 존재하면 로드
        if storage_path and os.path.exists(storage_path):
            self.load_from_file(storage_path)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """
        새 메시지 추가
        
        Args:
            role (str): 메시지 작성자의 역할 (user, assistant, system)
            content (str): 메시지 내용
            metadata (Optional[Dict[str, Any]], optional): 추가 메타데이터. 기본값은 None.
            
        Returns:
            ConversationMessage: 추가된 메시지
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.history.add_message(message)
        
        # 자동 저장이 활성화되어 있으면 파일에 저장
        if self.auto_save and self.storage_path:
            self.save_to_file(self.storage_path)
            
        return message
    
    def add_messages_batch(self, messages: List[Dict[str, Any]]) -> List[ConversationMessage]:
        """
        여러 메시지를 일괄 추가
        
        Args:
            messages (List[Dict[str, Any]]): 추가할 메시지 목록
            
        Returns:
            List[ConversationMessage]: 추가된 메시지 목록
        """
        added_messages = []
        for msg_data in messages:
            role = msg_data.get("role")
            content = msg_data.get("content")
            metadata = msg_data.get("metadata")
            
            if role and content:
                message = self.add_message(role, content, metadata)
                added_messages.append(message)
        
        return added_messages
    
    def get_context_window(self, dynamic_size: bool = False, include_system: bool = True, 
                       prioritize_important: bool = False) -> List[ConversationMessage]:
        """
        현재 컨텍스트 윈도우 반환 (최근 N개 메시지)
        
        Args:
            dynamic_size (bool, optional): 동적 크기 조정 사용 여부. 기본값은 False.
            include_system (bool, optional): 시스템 메시지 포함 여부. 기본값은 True.
            prioritize_important (bool, optional): 중요 메시지 우선 포함 여부. 기본값은 False.
            
        Returns:
            List[ConversationMessage]: 컨텍스트 윈도우에 포함된 메시지 목록
        """
        # 시스템 메시지 분리 (항상 포함)
        system_messages = []
        if include_system:
            system_messages = [msg for msg in self.history.messages if msg.role == "system"]
        
        # 비시스템 메시지 (user, assistant)
        non_system_messages = [msg for msg in self.history.messages if msg.role != "system"]
        
        # 기본 윈도우 크기 계산
        window_size = min(self.context_window_size, len(non_system_messages))
        if dynamic_size:
            # 동적 크기 계산 (복잡성에 따라 조정)
            window_size = min(self._calculate_dynamic_window_size(), len(non_system_messages))
        
        # 중요 메시지 우선 포함 여부에 따라 처리
        if not prioritize_important:
            # 단순히 최근 메시지 포함
            recent_messages = non_system_messages[-window_size:] if window_size > 0 else []
        else:
            # 중요 메시지 찾기
            important_messages = []
            for msg in non_system_messages:
                # 메타데이터에 importance가 있는 메시지 확인
                if msg.metadata and "importance" in msg.metadata:
                    important_messages.append(msg)
                    continue
                    
                # 중요 키워드가 있는 메시지 확인
                content = msg.content.lower()
                if "중요" in content or "중요한 질문" in content or "중요한 답변" in content or "꼭 기억" in content or "반드시" in content:
                    important_messages.append(msg)
                    continue
                    
                # 코드 블록이 있는 메시지 확인
                if "```" in content:
                    important_messages.append(msg)
            
            # 최근 메시지 (윈도우 크기의 1/3)
            recent_count = max(1, window_size // 3)
            recent_msgs = non_system_messages[-recent_count:]
            
            # 중요 메시지와 최근 메시지 결합 (중복 제거)
            msg_ids = set()
            combined_messages = []
            
            # 중요 메시지 추가 (중요 메시지를 우선적으로 포함)
            for msg in important_messages:
                if id(msg) not in msg_ids:
                    msg_ids.add(id(msg))
                    combined_messages.append(msg)
            
            # 최근 메시지 추가
            for msg in recent_msgs:
                if id(msg) not in msg_ids:
                    msg_ids.add(id(msg))
                    combined_messages.append(msg)
            
            # 시간순 정렬
            combined_messages.sort(key=lambda msg: msg.timestamp)
            
            # 윈도우 크기 제한 (중요 메시지가 있으면 최소 하나는 포함)
            if len(combined_messages) > window_size:
                # 중요 메시지가 있는지 확인
                has_important = False
                for msg in combined_messages[-window_size:]:
                    if msg in important_messages:
                        has_important = True
                        break
                
                # 중요 메시지가 없으면 하나 포함
                if not has_important and important_messages:
                    # 가장 최근 메시지 하나를 중요 메시지로 교체
                    result = combined_messages[-(window_size-1):] + [important_messages[-1]]
                    recent_messages = result
                else:
                    recent_messages = combined_messages[-window_size:]
            else:
                recent_messages = combined_messages
        
        # 시스템 메시지와 최근 메시지 결합
        # 시스템 메시지는 항상 컨텍스트의 시작 부분에 위치
        return system_messages + recent_messages
    
    def _calculate_dynamic_window_size(self) -> int:
        """
        대화 복잡성에 따른 동적 컨텍스트 윈도우 크기 계산
        
        Returns:
            int: 계산된 컨텍스트 윈도우 크기
        """
        # 기본 크기
        base_size = self.context_window_size
        
        # 메시지가 충분하지 않으면 기본 크기 반환
        if len(self.history.messages) <= base_size:
            return base_size
        
        # 최근 메시지 가져오기
        recent_messages = self.history.get_messages(min(20, len(self.history.messages)))
        
        # 복잡성 요소 계산
        # 1. 메시지 길이
        avg_length = sum(len(msg.content) for msg in recent_messages) / len(recent_messages)
        length_factor = 1.0
        if avg_length > 500:  # 긴 메시지
            length_factor = 0.8  # 메시지가 길면 더 적은 메시지 포함
        elif avg_length < 100:  # 짧은 메시지
            length_factor = 1.2  # 메시지가 짧으면 더 많은 메시지 포함
        
        # 2. 코드 블록 및 중요 키워드 확인
        code_blocks = sum(msg.content.count('```') for msg in recent_messages) // 2
        important_keywords = ["중요", "핵심", "필수", "기억", "remember", "key"]
        keyword_count = 0
        for msg in recent_messages:
            content = msg.content.lower()
            keyword_count += sum(content.count(keyword) for keyword in important_keywords)
        
        # 복잡성 점수 계산 (0.8 ~ 1.5)
        complexity_factor = 1.0
        if code_blocks > 0:
            complexity_factor += 0.2  # 코드 블록이 있으면 더 많은 컨텍스트 필요
        if keyword_count > 0:
            complexity_factor += 0.1  # 중요 키워드가 있으면 더 많은 컨텍스트 필요
        
        # 윈도우 크기 조정
        adjusted_size = min(
            max(
                base_size,
                int(base_size * complexity_factor * length_factor)
            ),
            self.max_memory_size
        )
        
        return adjusted_size
    
    def get_all_messages(self) -> List[ConversationMessage]:
        """
        모든 메시지 반환
        
        Returns:
            List[ConversationMessage]: 전체 메시지 목록
        """
        return self.history.get_messages()
    
    def get_messages_by_role(self, role: str) -> List[ConversationMessage]:
        """
        특정 역할의 메시지만 반환
        
        Args:
            role (str): 메시지 작성자의 역할 (user, assistant, system)
            
        Returns:
            List[ConversationMessage]: 지정된 역할의 메시지 목록
        """
        return self.history.get_messages_by_role(role)
    
    def clear_memory(self) -> None:
        """대화 메모리 초기화"""
        self.history.clear()
        
        # 자동 저장이 활성화되어 있으면 파일에 저장
        if self.auto_save and self.storage_path:
            self.save_to_file(self.storage_path)
    
    def set_context_window_size(self, size: int) -> None:
        """
        컨텍스트 윈도우 크기 설정
        
        Args:
            size (int): 새 컨텍스트 윈도우 크기
            
        Raises:
            ValueError: 크기가 1보다 작거나 최대 메모리 크기보다 큰 경우
        """
        if size < 1:
            raise ValueError("컨텍스트 윈도우 크기는 최소 1이어야 합니다.")
        if size > self.max_memory_size:
            raise ValueError(f"컨텍스트 윈도우 크기는 최대 메모리 크기({self.max_memory_size})를 초과할 수 없습니다.")
        
        self.context_window_size = size
    
    def set_max_memory_size(self, size: int) -> None:
        """
        최대 메모리 크기 설정
        
        Args:
            size (int): 새 최대 메모리 크기
            
        Raises:
            ValueError: 크기가 컨텍스트 윈도우 크기보다 작은 경우
        """
        if size < self.context_window_size:
            raise ValueError(f"최대 메모리 크기는 컨텍스트 윈도우 크기({self.context_window_size})보다 작을 수 없습니다.")
        
        self.max_memory_size = size
        self.history.max_messages = size
        
        # 메모리 크기가 줄어든 경우 초과 메시지 제거
        if len(self.history.messages) > size:
            self.history.messages = self.history.messages[-size:]
    
    def search_by_content(self, query: str, case_sensitive: bool = False) -> List[ConversationMessage]:
        """
        내용으로 메시지 검색
        
        Args:
            query (str): 검색할 문자열
            case_sensitive (bool, optional): 대소문자 구분 여부. 기본값은 False.
            
        Returns:
            List[ConversationMessage]: 검색 결과 메시지 목록
        """
        if not case_sensitive:
            query = query.lower()
            return [msg for msg in self.history.messages if query in msg.content.lower()]
        else:
            return [msg for msg in self.history.messages if query in msg.content]
    
    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        LLM 입력용 컨텍스트 형식으로 메시지 반환
        
        Returns:
            List[Dict[str, str]]: LLM 입력용 메시지 목록 (역할과 내용만 포함)
        """
        messages = self.get_context_window()
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        대화 메모리를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 직렬화된 대화 메모리
        """
        return {
            "context_window_size": self.context_window_size,
            "max_memory_size": self.max_memory_size,
            "messages": self.history.to_dict_list()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """
        딕셔너리에서 대화 메모리 생성
        
        Args:
            data (Dict[str, Any]): 직렬화된 대화 메모리
            
        Returns:
            ConversationMemory: 생성된 대화 메모리 객체
        """
        memory = cls(
            context_window_size=data.get("context_window_size", 10),
            max_memory_size=data.get("max_memory_size", 100)
        )
        
        # 메시지 추가
        messages_data = data.get("messages", [])
        for msg_data in messages_data:
            message = ConversationMessage.from_dict(msg_data)
            memory.history.add_message(message)
            
        return memory
    
    def save_to_file(self, file_path: str) -> bool:
        """
        대화 메모리를 파일에 저장
        
        Args:
            file_path (str): 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리 생성
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """
        파일에서 대화 메모리 로드
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 속성 설정
            self.context_window_size = data.get("context_window_size", self.context_window_size)
            self.max_memory_size = data.get("max_memory_size", self.max_memory_size)
            self.history.max_messages = self.max_memory_size
            
            # 메시지 로드
            self.history.clear()
            messages_data = data.get("messages", [])
            for msg_data in messages_data:
                message = ConversationMessage.from_dict(msg_data)
                self.history.add_message(message)
                
            return True
        except Exception as e:
            print(f"대화 기록 로드 중 오류 발생: {e}")
            return False