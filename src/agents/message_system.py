# src/agents/message_system.py

import asyncio
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentMessage, AgentResult


class MessagePriority(Enum):
    """메시지 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageStatus(Enum):
    """메시지 상태"""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MessageEnvelope:
    """메시지 봉투 - 메시지 전달을 위한 메타데이터 포함"""
    message: AgentMessage
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """메시지 만료 여부 확인"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def can_retry(self) -> bool:
        """재시도 가능 여부 확인"""
        return self.retry_count < self.max_retries and not self.is_expired()


class MessageQueue:
    """우선순위 기반 메시지 큐"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            MessagePriority.URGENT: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.NORMAL: deque(),
            MessagePriority.LOW: deque()
        }
        self.total_size = 0
        self.lock = asyncio.Lock()
    
    async def put(self, envelope: MessageEnvelope) -> bool:
        """메시지를 큐에 추가"""
        async with self.lock:
            if self.total_size >= self.max_size:
                # 큐가 가득 찬 경우 낮은 우선순위 메시지 제거
                if not self._remove_low_priority():
                    return False
            
            self.queues[envelope.priority].append(envelope)
            self.total_size += 1
            return True
    
    async def get(self) -> Optional[MessageEnvelope]:
        """우선순위 순으로 메시지 가져오기"""
        async with self.lock:
            for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                queue = self.queues[priority]
                if queue:
                    envelope = queue.popleft()
                    self.total_size -= 1
                    return envelope
            return None
    
    def _remove_low_priority(self) -> bool:
        """낮은 우선순위 메시지 제거"""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL]:
            queue = self.queues[priority]
            if queue:
                queue.popleft()
                self.total_size -= 1
                return True
        return False
    
    def size(self) -> int:
        """큐 크기 반환"""
        return self.total_size
    
    def is_empty(self) -> bool:
        """큐가 비어있는지 확인"""
        return self.total_size == 0


class MessageRouter:
    """메시지 라우팅 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("message_router")
        
        # 에이전트별 메시지 큐
        self.agent_queues: Dict[str, MessageQueue] = {}
        
        # 메시지 핸들러 등록
        self.message_handlers: Dict[str, Callable] = {}
        
        # 브로드캐스트 구독자
        self.broadcast_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # 메시지 추적
        self.message_tracking: Dict[str, MessageEnvelope] = {}
        
        # 라우팅 규칙
        self.routing_rules: Dict[str, Callable] = {}
        
        # 통계
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_expired": 0
        }
        
        self.logger.info("메시지 라우터 초기화 완료")
    
    def register_agent(self, agent_id: str, queue_size: int = 100):
        """에이전트를 라우터에 등록"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = MessageQueue(queue_size)
            self.logger.debug(f"에이전트 등록: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """에이전트를 라우터에서 제거"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
            # 브로드캐스트 구독에서도 제거
            for subscribers in self.broadcast_subscribers.values():
                subscribers.discard(agent_id)
            self.logger.debug(f"에이전트 제거: {agent_id}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """메시지 타입별 핸들러 등록"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"메시지 핸들러 등록: {message_type}")
    
    def add_routing_rule(self, rule_name: str, rule_func: Callable):
        """라우팅 규칙 추가"""
        self.routing_rules[rule_name] = rule_func
        self.logger.debug(f"라우팅 규칙 추가: {rule_name}")
    
    async def send_message(self, message: AgentMessage, 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          ttl_seconds: Optional[int] = None) -> bool:
        """메시지 전송"""
        try:
            # 메시지 봉투 생성
            envelope = MessageEnvelope(
                message=message,
                priority=priority,
                expires_at=datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            )
            
            # 메시지 추적 등록
            self.message_tracking[message.id] = envelope
            
            # 라우팅 규칙 적용
            target_agents = await self._apply_routing_rules(message)
            
            if not target_agents:
                target_agents = [message.receiver]
            
            # 각 대상 에이전트에게 전송
            success_count = 0
            for agent_id in target_agents:
                if await self._deliver_to_agent(agent_id, envelope):
                    success_count += 1
            
            self.stats["messages_sent"] += 1
            
            if success_count > 0:
                envelope.status = MessageStatus.DELIVERED
                envelope.delivered_at = datetime.now()
                self.stats["messages_delivered"] += 1
                return True
            else:
                envelope.status = MessageStatus.FAILED
                self.stats["messages_failed"] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"메시지 전송 중 오류: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def _deliver_to_agent(self, agent_id: str, envelope: MessageEnvelope) -> bool:
        """특정 에이전트에게 메시지 전달"""
        if agent_id not in self.agent_queues:
            self.logger.warning(f"등록되지 않은 에이전트: {agent_id}")
            return False
        
        queue = self.agent_queues[agent_id]
        success = await queue.put(envelope)
        
        if success:
            self.logger.debug(f"메시지 전달: {envelope.message.sender} -> {agent_id}")
        else:
            self.logger.warning(f"메시지 큐 가득참: {agent_id}")
        
        return success
    
    async def _apply_routing_rules(self, message: AgentMessage) -> List[str]:
        """라우팅 규칙 적용"""
        target_agents = []
        
        for rule_name, rule_func in self.routing_rules.items():
            try:
                agents = await rule_func(message)
                if agents:
                    target_agents.extend(agents)
            except Exception as e:
                self.logger.error(f"라우팅 규칙 적용 중 오류 ({rule_name}): {e}")
        
        return list(set(target_agents))  # 중복 제거
    
    async def get_message(self, agent_id: str) -> Optional[MessageEnvelope]:
        """에이전트의 메시지 큐에서 메시지 가져오기"""
        if agent_id not in self.agent_queues:
            return None
        
        queue = self.agent_queues[agent_id]
        envelope = await queue.get()
        
        if envelope:
            # 만료된 메시지 처리
            if envelope.is_expired():
                envelope.status = MessageStatus.EXPIRED
                self.stats["messages_expired"] += 1
                return None
            
            self.logger.debug(f"메시지 수신: {agent_id} <- {envelope.message.sender}")
        
        return envelope
    
    async def broadcast_message(self, sender_id: str, message_type: str, 
                              content: Dict[str, Any], 
                              priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """브로드캐스트 메시지 전송"""
        subscribers = self.broadcast_subscribers.get(message_type, set())
        
        if not subscribers:
            # 구독자가 없으면 모든 에이전트에게 전송
            subscribers = set(self.agent_queues.keys())
            subscribers.discard(sender_id)  # 발신자 제외
        
        sent_count = 0
        
        for agent_id in subscribers:
            message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=sender_id,
                receiver=agent_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now()
            )
            
            if await self.send_message(message, priority):
                sent_count += 1
        
        self.logger.info(f"브로드캐스트 완료: {message_type} -> {sent_count}개 에이전트")
        return sent_count
    
    def subscribe_to_broadcast(self, agent_id: str, message_type: str):
        """브로드캐스트 구독"""
        self.broadcast_subscribers[message_type].add(agent_id)
        self.logger.debug(f"브로드캐스트 구독: {agent_id} -> {message_type}")
    
    def unsubscribe_from_broadcast(self, agent_id: str, message_type: str):
        """브로드캐스트 구독 해제"""
        self.broadcast_subscribers[message_type].discard(agent_id)
        self.logger.debug(f"브로드캐스트 구독 해제: {agent_id} -> {message_type}")
    
    async def process_message_with_handler(self, envelope: MessageEnvelope) -> Optional[Any]:
        """메시지 핸들러로 메시지 처리"""
        message_type = envelope.message.message_type
        
        if message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
                result = await handler(envelope.message)
                
                envelope.status = MessageStatus.PROCESSED
                envelope.processed_at = datetime.now()
                
                return result
                
            except Exception as e:
                self.logger.error(f"메시지 핸들러 실행 중 오류: {e}")
                envelope.status = MessageStatus.FAILED
                return None
        else:
            self.logger.warning(f"등록되지 않은 메시지 타입: {message_type}")
            return None
    
    def get_queue_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """에이전트 큐 상태 조회"""
        if agent_id not in self.agent_queues:
            return None
        
        queue = self.agent_queues[agent_id]
        return {
            "agent_id": agent_id,
            "queue_size": queue.size(),
            "is_empty": queue.is_empty(),
            "max_size": queue.max_size
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        total_queue_size = sum(queue.size() for queue in self.agent_queues.values())
        
        return {
            "registered_agents": len(self.agent_queues),
            "total_queue_size": total_queue_size,
            "message_handlers": len(self.message_handlers),
            "routing_rules": len(self.routing_rules),
            "broadcast_types": len(self.broadcast_subscribers),
            "stats": self.stats.copy()
        }
    
    async def cleanup_expired_messages(self):
        """만료된 메시지 정리"""
        expired_count = 0
        
        for message_id, envelope in list(self.message_tracking.items()):
            if envelope.is_expired():
                envelope.status = MessageStatus.EXPIRED
                del self.message_tracking[message_id]
                expired_count += 1
        
        if expired_count > 0:
            self.stats["messages_expired"] += expired_count
            self.logger.info(f"만료된 메시지 {expired_count}개 정리")


class MessageSystem:
    """통합 메시지 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("message_system")
        self.router = MessageRouter()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5분마다 정리
        
        self.logger.info("메시지 시스템 초기화 완료")
    
    async def start(self):
        """메시지 시스템 시작"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("메시지 시스템 시작")
    
    async def stop(self):
        """메시지 시스템 정지"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("메시지 시스템 정지")
    
    async def _cleanup_loop(self):
        """정리 작업 루프"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.router.cleanup_expired_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"정리 작업 중 오류: {e}")
    
    def get_router(self) -> MessageRouter:
        """메시지 라우터 반환"""
        return self.router


# 전역 메시지 시스템 인스턴스
_global_message_system: Optional[MessageSystem] = None


def get_message_system() -> MessageSystem:
    """전역 메시지 시스템 인스턴스 반환"""
    global _global_message_system
    if _global_message_system is None:
        _global_message_system = MessageSystem()
    return _global_message_system