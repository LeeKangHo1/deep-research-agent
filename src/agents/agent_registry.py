# src/agents/agent_registry.py

from typing import Dict, List, Optional, Type, Any, Callable
import logging
import importlib
import inspect
from datetime import datetime
import asyncio
from pathlib import Path

from .base_agent import BaseAgent, AgentMessage, AgentResult


class AgentRegistry:
    """
    에이전트 레지스트리 - 모든 에이전트를 관리하는 중앙 시스템
    
    에이전트 등록, 조회, 생성, 메시지 라우팅 등의 기능을 제공합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("agent_registry")
        
        # 등록된 에이전트 클래스들
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        
        # 활성 에이전트 인스턴스들
        self._active_agents: Dict[str, BaseAgent] = {}
        
        # 에이전트 설정
        self._agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # 메시지 라우팅 테이블
        self._message_handlers: Dict[str, Callable] = {}
        
        # 에이전트 생성 팩토리 함수들
        self._agent_factories: Dict[str, Callable] = {}
        
        self.logger.info("에이전트 레지스트리 초기화 완료")
    
    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent], 
                           config: Optional[Dict[str, Any]] = None):
        """
        에이전트 클래스를 레지스트리에 등록
        
        Args:
            agent_type: 에이전트 타입 식별자
            agent_class: 에이전트 클래스
            config: 기본 설정
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class}는 BaseAgent를 상속받아야 합니다")
        
        self._agent_classes[agent_type] = agent_class
        self._agent_configs[agent_type] = config or {}
        
        self.logger.info(f"에이전트 클래스 등록: {agent_type} -> {agent_class.__name__}")
    
    def register_agent_factory(self, agent_type: str, factory_func: Callable):
        """
        에이전트 생성 팩토리 함수 등록
        
        Args:
            agent_type: 에이전트 타입
            factory_func: 팩토리 함수
        """
        self._agent_factories[agent_type] = factory_func
        self.logger.info(f"에이전트 팩토리 등록: {agent_type}")
    
    def get_registered_types(self) -> List[str]:
        """등록된 에이전트 타입 목록 반환"""
        return list(self._agent_classes.keys())
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """에이전트 타입으로 클래스 조회"""
        return self._agent_classes.get(agent_type)
    
    async def create_agent(self, agent_type: str, agent_id: str, 
                          config: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        """
        새로운 에이전트 인스턴스 생성
        
        Args:
            agent_type: 에이전트 타입
            agent_id: 에이전트 고유 ID
            config: 에이전트별 설정
            
        Returns:
            Optional[BaseAgent]: 생성된 에이전트 인스턴스
        """
        try:
            # 팩토리 함수가 있으면 우선 사용
            if agent_type in self._agent_factories:
                factory = self._agent_factories[agent_type]
                agent = factory(agent_id, config)
            else:
                # 클래스로 직접 생성
                agent_class = self._agent_classes.get(agent_type)
                if not agent_class:
                    self.logger.error(f"등록되지 않은 에이전트 타입: {agent_type}")
                    return None
                
                # 기본 설정과 사용자 설정 병합
                merged_config = self._agent_configs.get(agent_type, {}).copy()
                if config:
                    merged_config.update(config)
                
                # 에이전트 생성
                agent = agent_class(
                    agent_id=agent_id,
                    name=f"{agent_type}_{agent_id}",
                    description=f"{agent_type} 에이전트"
                )
                
                # 설정 적용
                agent.set_config(merged_config)
            
            # 리소스 초기화
            if await agent.initialize_resources():
                self._active_agents[agent_id] = agent
                agent.update_status("ready")
                self.logger.info(f"에이전트 생성 완료: {agent_id} ({agent_type})")
                return agent
            else:
                self.logger.error(f"에이전트 리소스 초기화 실패: {agent_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"에이전트 생성 중 오류: {e}")
            return None
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """에이전트 ID로 활성 에이전트 조회"""
        return self._active_agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """모든 활성 에이전트 반환"""
        return self._active_agents.copy()
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """특정 타입의 모든 에이전트 반환"""
        return [agent for agent in self._active_agents.values() 
                if agent.__class__.__name__.lower().startswith(agent_type.lower())]
    
    def get_available_agents(self) -> List[BaseAgent]:
        """사용 가능한 에이전트 목록 반환"""
        return [agent for agent in self._active_agents.values() 
                if agent.is_available()]
    
    async def remove_agent(self, agent_id: str) -> bool:
        """
        에이전트 제거
        
        Args:
            agent_id: 제거할 에이전트 ID
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            agent = self._active_agents.get(agent_id)
            if agent:
                await agent.stop()
                del self._active_agents[agent_id]
                self.logger.info(f"에이전트 제거 완료: {agent_id}")
                return True
            else:
                self.logger.warning(f"존재하지 않는 에이전트: {agent_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"에이전트 제거 중 오류: {e}")
            return False
    
    async def route_message(self, message: AgentMessage) -> bool:
        """
        메시지를 대상 에이전트에게 라우팅
        
        Args:
            message: 라우팅할 메시지
            
        Returns:
            bool: 라우팅 성공 여부
        """
        try:
            target_agent = self._active_agents.get(message.receiver)
            if target_agent:
                target_agent.add_message(message)
                self.logger.debug(f"메시지 라우팅: {message.sender} -> {message.receiver}")
                return True
            else:
                self.logger.warning(f"대상 에이전트를 찾을 수 없음: {message.receiver}")
                return False
                
        except Exception as e:
            self.logger.error(f"메시지 라우팅 중 오류: {e}")
            return False
    
    async def broadcast_message(self, sender_id: str, message_type: str, 
                              content: Dict[str, Any], exclude_sender: bool = True) -> int:
        """
        모든 에이전트에게 메시지 브로드캐스트
        
        Args:
            sender_id: 발신자 ID
            message_type: 메시지 타입
            content: 메시지 내용
            exclude_sender: 발신자 제외 여부
            
        Returns:
            int: 메시지를 받은 에이전트 수
        """
        sent_count = 0
        
        for agent_id, agent in self._active_agents.items():
            if exclude_sender and agent_id == sender_id:
                continue
            
            try:
                message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender=sender_id,
                    receiver=agent_id,
                    message_type=message_type,
                    content=content,
                    timestamp=datetime.now()
                )
                
                agent.add_message(message)
                sent_count += 1
                
            except Exception as e:
                self.logger.error(f"브로드캐스트 중 오류 ({agent_id}): {e}")
        
        self.logger.info(f"브로드캐스트 완료: {sent_count}개 에이전트")
        return sent_count
    
    def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        특정 능력을 가진 에이전트들 검색
        
        Args:
            capability: 검색할 능력
            
        Returns:
            List[BaseAgent]: 해당 능력을 가진 에이전트 목록
        """
        matching_agents = []
        
        for agent in self._active_agents.values():
            if agent.can_handle_task(capability):
                matching_agents.append(agent)
        
        return matching_agents
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        전체 시스템 상태 반환
        
        Returns:
            Dict[str, Any]: 시스템 상태 정보
        """
        total_agents = len(self._active_agents)
        available_agents = len(self.get_available_agents())
        
        agent_status_counts = {}
        for agent in self._active_agents.values():
            status = agent.status
            agent_status_counts[status] = agent_status_counts.get(status, 0) + 1
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "registered_types": len(self._agent_classes),
            "agent_status_counts": agent_status_counts,
            "timestamp": datetime.now()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        모든 에이전트의 상태 확인
        
        Returns:
            Dict[str, Any]: 상태 확인 결과
        """
        health_results = {}
        
        for agent_id, agent in self._active_agents.items():
            try:
                is_healthy = await agent.health_check()
                health_results[agent_id] = {
                    "healthy": is_healthy,
                    "status": agent.status,
                    "last_activity": agent.last_activity
                }
            except Exception as e:
                health_results[agent_id] = {
                    "healthy": False,
                    "status": "error",
                    "error": str(e)
                }
        
        return health_results
    
    async def shutdown_all(self):
        """모든 에이전트 종료"""
        self.logger.info("모든 에이전트 종료 시작")
        
        shutdown_tasks = []
        for agent in self._active_agents.values():
            shutdown_tasks.append(agent.stop())
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self._active_agents.clear()
        self.logger.info("모든 에이전트 종료 완료")


# 전역 에이전트 레지스트리 인스턴스
_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """전역 에이전트 레지스트리 인스턴스 반환"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(agent_type: str, config: Optional[Dict[str, Any]] = None):
    """
    에이전트 클래스 등록을 위한 데코레이터
    
    Args:
        agent_type: 에이전트 타입
        config: 기본 설정
    """
    def decorator(agent_class: Type[BaseAgent]):
        registry = get_agent_registry()
        registry.register_agent_class(agent_type, agent_class, config)
        return agent_class
    return decorator