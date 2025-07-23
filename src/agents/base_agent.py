# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import uuid
import asyncio

from ..models.research_models import ResearchData, AnalysisResult, SynthesisResult
from ..utils.state_manager import StateManager


@dataclass
class AgentMessage:
    """에이전트 간 메시지 전달을 위한 데이터 클래스"""
    id: str
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """에이전트 작업 결과를 위한 데이터 클래스"""
    agent_id: str
    task_id: str
    status: str  # 'success', 'failure', 'partial'
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseAgent(ABC):
    """
    모든 에이전트의 기본 추상 클래스
    
    모든 전문 에이전트는 이 클래스를 상속받아 구현해야 합니다.
    공통 인터페이스와 기본 기능을 제공합니다.
    """
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        """
        BaseAgent 초기화
        
        Args:
            agent_id: 에이전트 고유 식별자
            name: 에이전트 이름
            description: 에이전트 설명
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = "initialized"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # 로깅 설정
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        
        # 상태 관리자 참조
        self.state_manager: Optional[StateManager] = None
        
        # 메시지 시스템 참조
        self.message_system = None
        
        # 결과 공유 시스템 참조
        self.result_sharing_system = None
        
        # 메시지 큐 (에이전트 간 통신용)
        self.message_queue: List[AgentMessage] = []
        
        # 작업 기록
        self.task_history: List[Dict[str, Any]] = []
        
        # 에이전트 설정
        self.config: Dict[str, Any] = {}
        
        self.logger.info(f"에이전트 {self.name} ({self.agent_id}) 초기화 완료")
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """
        에이전트의 주요 작업을 실행하는 추상 메서드
        
        Args:
            task: 실행할 작업 정보
            
        Returns:
            AgentResult: 작업 실행 결과
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        에이전트가 수행할 수 있는 작업 목록을 반환
        
        Returns:
            List[str]: 수행 가능한 작업 목록
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentResult]:
        """
        다른 에이전트로부터 받은 메시지를 처리
        
        Args:
            message: 처리할 메시지
            
        Returns:
            Optional[AgentResult]: 메시지 처리 결과 (필요한 경우)
        """
        pass
    
    @abstractmethod
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        주어진 작업이 이 에이전트가 처리할 수 있는지 검증
        
        Args:
            task: 검증할 작업
            
        Returns:
            bool: 처리 가능 여부
        """
        pass
    
    @abstractmethod
    async def initialize_resources(self) -> bool:
        """
        에이전트가 작업을 수행하기 위해 필요한 리소스 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        pass
    
    @abstractmethod
    async def cleanup_resources(self):
        """
        에이전트 종료 시 리소스 정리
        """
        pass
    
    def set_state_manager(self, state_manager: StateManager):
        """상태 관리자 설정"""
        self.state_manager = state_manager
        self.logger.debug("상태 관리자 설정 완료")
    
    def set_message_system(self, message_system):
        """메시지 시스템 설정"""
        self.message_system = message_system
        # 메시지 라우터에 에이전트 등록
        if message_system:
            message_system.get_router().register_agent(self.agent_id)
        self.logger.debug("메시지 시스템 설정 완료")
    
    def set_result_sharing_system(self, result_sharing_system):
        """결과 공유 시스템 설정"""
        self.result_sharing_system = result_sharing_system
        self.logger.debug("결과 공유 시스템 설정 완료")
    
    def update_status(self, status: str):
        """에이전트 상태 업데이트"""
        old_status = self.status
        self.status = status
        self.last_activity = datetime.now()
        self.logger.debug(f"상태 변경: {old_status} -> {status}")
    
    def add_message(self, message: AgentMessage):
        """메시지 큐에 메시지 추가"""
        self.message_queue.append(message)
        self.logger.debug(f"메시지 수신: {message.sender} -> {message.message_type}")
    
    def get_messages(self, message_type: Optional[str] = None) -> List[AgentMessage]:
        """메시지 큐에서 메시지 조회"""
        if message_type:
            return [msg for msg in self.message_queue if msg.message_type == message_type]
        return self.message_queue.copy()
    
    def clear_messages(self):
        """메시지 큐 초기화"""
        cleared_count = len(self.message_queue)
        self.message_queue.clear()
        self.logger.debug(f"{cleared_count}개 메시지 삭제")
    
    def create_message(self, receiver: str, message_type: str, content: Dict[str, Any], 
                      metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """새 메시지 생성"""
        return AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def log_task(self, task_id: str, task_type: str, status: str, 
                details: Optional[Dict[str, Any]] = None):
        """작업 기록 추가"""
        task_record = {
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "timestamp": datetime.now(),
            "details": details or {}
        }
        self.task_history.append(task_record)
        self.logger.info(f"작업 기록: {task_type} - {status}")
    
    def get_task_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """작업 기록 조회"""
        if limit:
            return self.task_history[-limit:]
        return self.task_history.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """에이전트 설정 업데이트"""
        self.config.update(config)
        self.logger.debug("설정 업데이트 완료")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        return self.config.get(key, default)
    
    def get_status_info(self) -> Dict[str, Any]:
        """에이전트 상태 정보 반환"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": len(self.message_queue),
            "task_count": len(self.task_history),
            "capabilities": self.get_capabilities()
        }
    
    async def health_check(self) -> bool:
        """에이전트 상태 확인"""
        try:
            # 기본 상태 확인 로직
            if self.status == "error":
                return False
            
            # 상태 관리자 연결 확인
            if self.state_manager is None:
                self.logger.warning("상태 관리자가 설정되지 않음")
            
            return True
        except Exception as e:
            self.logger.error(f"상태 확인 중 오류: {e}")
            return False
    
    def __str__(self) -> str:
        return f"Agent({self.name}, {self.agent_id}, {self.status})"
    
    def __repr__(self) -> str:
        return (f"BaseAgent(agent_id='{self.agent_id}', name='{self.name}', "
                f"status='{self.status}', capabilities={len(self.get_capabilities())})")
    
    # 공통 인터페이스 메서드들
    
    async def send_message(self, receiver: str, message_type: str, content: Dict[str, Any],
                          callback: Optional[Callable] = None) -> bool:
        """
        다른 에이전트에게 메시지 전송
        
        Args:
            receiver: 수신자 에이전트 ID
            message_type: 메시지 타입
            content: 메시지 내용
            callback: 응답 처리 콜백 함수
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = self.create_message(receiver, message_type, content)
            
            # 메시지 시스템을 통해 메시지 전달
            if self.message_system:
                router = self.message_system.get_router()
                success = await router.send_message(message)
                if success:
                    self.logger.debug(f"메시지 전송 성공: {receiver}")
                    return True
            
            # 폴백: 상태 관리자를 통해 메시지 전달
            elif self.state_manager:
                success = await self.state_manager.route_message(message)
                if success:
                    self.logger.debug(f"메시지 전송 성공: {receiver}")
                    return True
            
            self.logger.warning(f"메시지 전송 실패: {receiver}")
            return False
            
        except Exception as e:
            self.logger.error(f"메시지 전송 중 오류: {e}")
            return False
    
    async def request_collaboration(self, target_agent: str, task_type: str, 
                                  data: Dict[str, Any]) -> Optional[AgentResult]:
        """
        다른 에이전트와의 협력 요청
        
        Args:
            target_agent: 협력 대상 에이전트 ID
            task_type: 요청할 작업 타입
            data: 작업 데이터
            
        Returns:
            Optional[AgentResult]: 협력 결과
        """
        try:
            collaboration_request = {
                "task_type": task_type,
                "data": data,
                "requester": self.agent_id,
                "timestamp": datetime.now()
            }
            
            success = await self.send_message(
                target_agent, 
                "collaboration_request", 
                collaboration_request
            )
            
            if success:
                self.logger.info(f"협력 요청 전송: {target_agent} - {task_type}")
                # 응답 대기 로직은 상위 오케스트레이터에서 처리
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"협력 요청 중 오류: {e}")
            return None
    
    async def share_result(self, result: AgentResult, result_type=None, 
                          target_agents: Optional[List[str]] = None,
                          scope=None, ttl_hours: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        작업 결과를 다른 에이전트들과 공유
        
        Args:
            result: 공유할 결과
            result_type: 결과 타입
            target_agents: 대상 에이전트 목록 (None이면 모든 에이전트)
            scope: 공유 범위
            ttl_hours: 만료 시간 (시간)
            metadata: 추가 메타데이터
        """
        try:
            # 결과 공유 시스템을 통한 공유
            if self.result_sharing_system and result_type:
                from .result_sharing import ResultType, ShareScope
                
                # 기본값 설정
                if scope is None:
                    scope = ShareScope.PRIVATE if target_agents else ShareScope.PUBLIC
                
                result_id = await self.result_sharing_system.share_result(
                    result=result,
                    result_type=result_type,
                    scope=scope,
                    target_agents=target_agents,
                    ttl_hours=ttl_hours,
                    metadata=metadata
                )
                
                self.logger.info(f"결과 공유 완료 (공유 시스템): {result_id}")
                return result_id
            
            # 폴백: 메시지 시스템을 통한 공유
            share_content = {
                "result": result,
                "shared_by": self.agent_id,
                "timestamp": datetime.now(),
                "metadata": metadata or {}
            }
            
            if target_agents:
                for agent_id in target_agents:
                    await self.send_message(agent_id, "result_share", share_content)
            else:
                # 브로드캐스트
                if self.message_system:
                    router = self.message_system.get_router()
                    await router.broadcast_message(
                        self.agent_id, "result_share", share_content
                    )
                elif self.state_manager:
                    await self.state_manager.broadcast_message(
                        self.agent_id, "result_share", share_content
                    )
            
            self.logger.info("결과 공유 완료 (메시지 시스템)")
            return None
            
        except Exception as e:
            self.logger.error(f"결과 공유 중 오류: {e}")
            return None
    
    def can_handle_task(self, task_type: str) -> bool:
        """
        특정 작업 타입을 처리할 수 있는지 확인
        
        Args:
            task_type: 작업 타입
            
        Returns:
            bool: 처리 가능 여부
        """
        return task_type in self.get_capabilities()
    
    async def get_workload_status(self) -> Dict[str, Any]:
        """
        현재 작업 부하 상태 반환
        
        Returns:
            Dict[str, Any]: 작업 부하 정보
        """
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "queue_size": len(self.message_queue),
            "active_tasks": len([t for t in self.task_history 
                               if t.get("status") == "running"]),
            "last_activity": self.last_activity,
            "availability": self.status in ["idle", "ready"]
        }
    
    async def pause(self):
        """에이전트 일시 정지"""
        if self.status not in ["error", "stopped"]:
            self.update_status("paused")
            self.logger.info("에이전트 일시 정지")
    
    async def resume(self):
        """에이전트 재개"""
        if self.status == "paused":
            self.update_status("ready")
            self.logger.info("에이전트 재개")
    
    async def stop(self):
        """에이전트 정지"""
        self.update_status("stopping")
        await self.cleanup_resources()
        self.update_status("stopped")
        self.logger.info("에이전트 정지 완료")
    
    # 기본 메서드 구현
    
    async def execute_with_retry(self, task: Dict[str, Any], max_retries: int = 3) -> AgentResult:
        """
        재시도 로직을 포함한 작업 실행
        
        Args:
            task: 실행할 작업
            max_retries: 최대 재시도 횟수
            
        Returns:
            AgentResult: 작업 실행 결과
        """
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"작업 실행 시도 {attempt + 1}/{max_retries + 1}: {task_id}")
                
                # 작업 검증
                if not self.validate_task(task):
                    return AgentResult(
                        agent_id=self.agent_id,
                        task_id=task_id,
                        status="failure",
                        result=None,
                        error="작업 검증 실패"
                    )
                
                # 작업 실행
                result = await self.execute(task)
                
                if result.status == "success":
                    self.log_task(task_id, task.get("type", "unknown"), "completed")
                    return result
                
                # 부분 성공인 경우 재시도하지 않음
                if result.status == "partial":
                    self.log_task(task_id, task.get("type", "unknown"), "partial")
                    return result
                
                # 실패한 경우 재시도
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 지수 백오프
                    self.logger.warning(f"작업 실패, {wait_time}초 후 재시도: {result.error}")
                    await asyncio.sleep(wait_time)
                else:
                    self.log_task(task_id, task.get("type", "unknown"), "failed", 
                                {"error": result.error, "attempts": attempt + 1})
                    return result
                    
            except Exception as e:
                error_msg = f"작업 실행 중 예외 발생: {str(e)}"
                self.logger.error(error_msg)
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    self.log_task(task_id, task.get("type", "unknown"), "error", 
                                {"error": error_msg, "attempts": attempt + 1})
                    return AgentResult(
                        agent_id=self.agent_id,
                        task_id=task_id,
                        status="failure",
                        result=None,
                        error=error_msg
                    )
        
        # 이 지점에 도달하면 안 됨
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task_id,
            status="failure",
            result=None,
            error="알 수 없는 오류"
        )
    
    async def execute_with_timeout(self, task: Dict[str, Any], timeout_seconds: int = 300) -> AgentResult:
        """
        타임아웃을 적용한 작업 실행
        
        Args:
            task: 실행할 작업
            timeout_seconds: 타임아웃 시간 (초)
            
        Returns:
            AgentResult: 작업 실행 결과
        """
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        try:
            result = await asyncio.wait_for(
                self.execute_with_retry(task), 
                timeout=timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"작업 타임아웃 ({timeout_seconds}초)"
            self.logger.error(error_msg)
            self.log_task(task_id, task.get("type", "unknown"), "timeout")
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task_id,
                status="failure",
                result=None,
                error=error_msg
            )
    
    def measure_performance(self, func_name: str):
        """
        성능 측정을 위한 데코레이터 팩토리
        
        Args:
            func_name: 측정할 함수 이름
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.debug(f"성능 측정 - {func_name}: {duration:.2f}초")
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.error(f"성능 측정 - {func_name} 실패 ({duration:.2f}초): {e}")
                    raise
            return wrapper
        return decorator
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        에러 처리 공통 로직
        
        Args:
            error: 발생한 에러
            context: 에러 발생 컨텍스트
            
        Returns:
            bool: 복구 가능 여부
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now(),
            "agent_id": self.agent_id
        }
        
        self.logger.error(f"에러 발생: {error_info}")
        
        # 에러 타입별 처리
        if isinstance(error, ConnectionError):
            self.logger.warning("연결 오류 - 재시도 가능")
            return True
        elif isinstance(error, TimeoutError):
            self.logger.warning("타임아웃 오류 - 재시도 가능")
            return True
        elif isinstance(error, ValueError):
            self.logger.error("값 오류 - 재시도 불가능")
            return False
        else:
            self.logger.error("알 수 없는 오류 - 재시도 불가능")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        에이전트 성능 메트릭 반환
        
        Returns:
            Dict[str, Any]: 성능 메트릭
        """
        total_tasks = len(self.task_history)
        successful_tasks = len([t for t in self.task_history if t.get("status") == "completed"])
        failed_tasks = len([t for t in self.task_history if t.get("status") in ["failed", "error"]])
        
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "agent_id": self.agent_id,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": round(success_rate, 2),
            "current_status": self.status,
            "uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_activity": self.last_activity,
            "message_queue_size": len(self.message_queue)
        }
    
    async def reset(self):
        """
        에이전트 상태 초기화
        """
        self.logger.info("에이전트 상태 초기화 시작")
        
        # 메시지 큐 초기화
        self.clear_messages()
        
        # 상태 초기화
        self.update_status("initialized")
        
        # 리소스 정리 후 재초기화
        await self.cleanup_resources()
        success = await self.initialize_resources()
        
        if success:
            self.update_status("ready")
            self.logger.info("에이전트 초기화 완료")
        else:
            self.update_status("error")
            self.logger.error("에이전트 초기화 실패")
    
    def is_available(self) -> bool:
        """
        에이전트가 새로운 작업을 받을 수 있는 상태인지 확인
        
        Returns:
            bool: 사용 가능 여부
        """
        return self.status in ["ready", "idle"] and len(self.message_queue) < 100
    
    # 결과 공유 시스템 관련 메서드들
    
    async def search_shared_results(self, result_type=None, keywords: Optional[List[str]] = None,
                                   tags: Optional[List[str]] = None, 
                                   shared_by: Optional[str] = None) -> List[str]:
        """
        공유된 결과 검색
        
        Args:
            result_type: 결과 타입 필터
            keywords: 키워드 필터
            tags: 태그 필터
            shared_by: 공유자 필터
            
        Returns:
            List[str]: 검색된 결과 ID 목록
        """
        if not self.result_sharing_system:
            self.logger.warning("결과 공유 시스템이 설정되지 않음")
            return []
        
        try:
            return await self.result_sharing_system.search_results(
                requester_id=self.agent_id,
                result_type=result_type,
                keywords=keywords,
                tags=tags,
                shared_by=shared_by
            )
        except Exception as e:
            self.logger.error(f"결과 검색 중 오류: {e}")
            return []
    
    async def get_shared_result(self, result_id: str) -> Optional[AgentResult]:
        """
        공유된 결과 조회
        
        Args:
            result_id: 결과 ID
            
        Returns:
            Optional[AgentResult]: 조회된 결과
        """
        if not self.result_sharing_system:
            self.logger.warning("결과 공유 시스템이 설정되지 않음")
            return None
        
        try:
            return await self.result_sharing_system.get_result(result_id, self.agent_id)
        except Exception as e:
            self.logger.error(f"결과 조회 중 오류: {e}")
            return None
    
    def subscribe_to_result_type(self, result_type):
        """결과 타입 구독"""
        if self.result_sharing_system:
            self.result_sharing_system.subscribe_to_results(self.agent_id, result_type)
            self.logger.debug(f"결과 타입 구독: {result_type}")
    
    def subscribe_to_keyword(self, keyword: str):
        """키워드 구독"""
        if self.result_sharing_system:
            self.result_sharing_system.subscribe_to_keyword(self.agent_id, keyword)
            self.logger.debug(f"키워드 구독: {keyword}")
    
    def join_group(self, group_name: str):
        """그룹 참여"""
        if self.result_sharing_system:
            self.result_sharing_system.add_agent_to_group(self.agent_id, group_name)
            self.logger.debug(f"그룹 참여: {group_name}")
    
    def leave_group(self, group_name: str):
        """그룹 탈퇴"""
        if self.result_sharing_system:
            self.result_sharing_system.remove_agent_from_group(self.agent_id, group_name)
            self.logger.debug(f"그룹 탈퇴: {group_name}")