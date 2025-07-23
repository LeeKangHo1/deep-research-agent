# src/agents/result_sharing.py

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import asyncio
from collections import defaultdict
import json
import pickle

from .base_agent import BaseAgent, AgentResult, AgentMessage


class ResultType(Enum):
    """결과 타입"""
    RESEARCH_DATA = "research_data"
    ANALYSIS_RESULT = "analysis_result"
    SYNTHESIS_RESULT = "synthesis_result"
    VALIDATION_RESULT = "validation_result"
    INTERMEDIATE_RESULT = "intermediate_result"
    FINAL_RESULT = "final_result"


class ShareScope(Enum):
    """공유 범위"""
    PRIVATE = "private"          # 특정 에이전트만
    GROUP = "group"              # 특정 그룹
    PUBLIC = "public"            # 모든 에이전트
    WORKFLOW = "workflow"        # 같은 워크플로우 내


@dataclass
class SharedResult:
    """공유된 결과 데이터"""
    id: str
    result: AgentResult
    result_type: ResultType
    shared_by: str
    shared_at: datetime
    scope: ShareScope
    target_agents: Optional[Set[str]] = None
    target_groups: Optional[Set[str]] = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """결과 만료 여부 확인"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def can_access(self, agent_id: str, agent_groups: Optional[Set[str]] = None) -> bool:
        """접근 권한 확인"""
        if self.is_expired():
            return False
        
        if self.scope == ShareScope.PUBLIC:
            return True
        elif self.scope == ShareScope.PRIVATE:
            return self.target_agents and agent_id in self.target_agents
        elif self.scope == ShareScope.GROUP:
            if not agent_groups or not self.target_groups:
                return False
            return bool(agent_groups.intersection(self.target_groups))
        elif self.scope == ShareScope.WORKFLOW:
            # 워크플로우 기반 접근 제어는 별도 구현 필요
            return True
        
        return False


class ResultIndex:
    """결과 인덱싱 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("result_index")
        
        # 타입별 인덱스
        self.type_index: Dict[ResultType, Set[str]] = defaultdict(set)
        
        # 에이전트별 인덱스
        self.agent_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 시간별 인덱스
        self.time_index: Dict[str, Set[str]] = defaultdict(set)  # YYYY-MM-DD 형식
        
        # 키워드 인덱스
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 태그 인덱스
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_result(self, shared_result: SharedResult):
        """결과를 인덱스에 추가"""
        result_id = shared_result.id
        
        # 타입별 인덱스
        self.type_index[shared_result.result_type].add(result_id)
        
        # 에이전트별 인덱스
        self.agent_index[shared_result.shared_by].add(result_id)
        
        # 시간별 인덱스
        date_key = shared_result.shared_at.strftime("%Y-%m-%d")
        self.time_index[date_key].add(result_id)
        
        # 메타데이터에서 키워드와 태그 추출
        metadata = shared_result.metadata
        
        # 키워드 인덱싱
        keywords = metadata.get("keywords", [])
        for keyword in keywords:
            self.keyword_index[keyword.lower()].add(result_id)
        
        # 태그 인덱싱
        tags = metadata.get("tags", [])
        for tag in tags:
            self.tag_index[tag.lower()].add(result_id)
        
        self.logger.debug(f"결과 인덱스 추가: {result_id}")
    
    def remove_result(self, result_id: str, shared_result: SharedResult):
        """결과를 인덱스에서 제거"""
        # 모든 인덱스에서 제거
        self.type_index[shared_result.result_type].discard(result_id)
        self.agent_index[shared_result.shared_by].discard(result_id)
        
        date_key = shared_result.shared_at.strftime("%Y-%m-%d")
        self.time_index[date_key].discard(result_id)
        
        # 키워드와 태그에서도 제거
        keywords = shared_result.metadata.get("keywords", [])
        for keyword in keywords:
            self.keyword_index[keyword.lower()].discard(result_id)
        
        tags = shared_result.metadata.get("tags", [])
        for tag in tags:
            self.tag_index[tag.lower()].discard(result_id)
        
        self.logger.debug(f"결과 인덱스 제거: {result_id}")
    
    def search_by_type(self, result_type: ResultType) -> Set[str]:
        """타입으로 결과 검색"""
        return self.type_index[result_type].copy()
    
    def search_by_agent(self, agent_id: str) -> Set[str]:
        """에이전트로 결과 검색"""
        return self.agent_index[agent_id].copy()
    
    def search_by_keyword(self, keyword: str) -> Set[str]:
        """키워드로 결과 검색"""
        return self.keyword_index[keyword.lower()].copy()
    
    def search_by_tag(self, tag: str) -> Set[str]:
        """태그로 결과 검색"""
        return self.tag_index[tag.lower()].copy()
    
    def search_by_date_range(self, start_date: datetime, end_date: datetime) -> Set[str]:
        """날짜 범위로 결과 검색"""
        result_ids = set()
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            date_key = current_date.strftime("%Y-%m-%d")
            result_ids.update(self.time_index[date_key])
            current_date += timedelta(days=1)
        
        return result_ids


class ResultSharingSystem:
    """결과 공유 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("result_sharing")
        
        # 공유된 결과 저장소
        self.shared_results: Dict[str, SharedResult] = {}
        
        # 결과 인덱스
        self.index = ResultIndex()
        
        # 에이전트 그룹 정보
        self.agent_groups: Dict[str, Set[str]] = defaultdict(set)
        
        # 구독 시스템
        self.subscribers: Dict[ResultType, Set[str]] = defaultdict(set)
        self.keyword_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # 접근 로그
        self.access_log: List[Dict[str, Any]] = []
        
        # 통계
        self.stats = {
            "results_shared": 0,
            "results_accessed": 0,
            "results_expired": 0
        }
        
        self.logger.info("결과 공유 시스템 초기화 완료")
    
    async def share_result(self, result: AgentResult, result_type: ResultType,
                          scope: ShareScope = ShareScope.PUBLIC,
                          target_agents: Optional[List[str]] = None,
                          target_groups: Optional[List[str]] = None,
                          ttl_hours: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        결과 공유
        
        Args:
            result: 공유할 결과
            result_type: 결과 타입
            scope: 공유 범위
            target_agents: 대상 에이전트 목록
            target_groups: 대상 그룹 목록
            ttl_hours: 만료 시간 (시간)
            metadata: 추가 메타데이터
            
        Returns:
            str: 공유 결과 ID
        """
        try:
            shared_result = SharedResult(
                id=str(uuid.uuid4()),
                result=result,
                result_type=result_type,
                shared_by=result.agent_id,
                shared_at=datetime.now(),
                scope=scope,
                target_agents=set(target_agents) if target_agents else None,
                target_groups=set(target_groups) if target_groups else None,
                expires_at=datetime.now() + timedelta(hours=ttl_hours) if ttl_hours else None,
                metadata=metadata or {}
            )
            
            # 저장소에 추가
            self.shared_results[shared_result.id] = shared_result
            
            # 인덱스에 추가
            self.index.add_result(shared_result)
            
            # 구독자들에게 알림
            await self._notify_subscribers(shared_result)
            
            self.stats["results_shared"] += 1
            self.logger.info(f"결과 공유 완료: {shared_result.id} ({result_type.value})")
            
            return shared_result.id
            
        except Exception as e:
            self.logger.error(f"결과 공유 중 오류: {e}")
            raise
    
    async def get_result(self, result_id: str, requester_id: str) -> Optional[AgentResult]:
        """
        공유된 결과 조회
        
        Args:
            result_id: 결과 ID
            requester_id: 요청자 에이전트 ID
            
        Returns:
            Optional[AgentResult]: 조회된 결과
        """
        try:
            shared_result = self.shared_results.get(result_id)
            if not shared_result:
                return None
            
            # 접근 권한 확인
            requester_groups = self.agent_groups.get(requester_id, set())
            if not shared_result.can_access(requester_id, requester_groups):
                self.logger.warning(f"결과 접근 권한 없음: {requester_id} -> {result_id}")
                return None
            
            # 접근 카운트 증가
            shared_result.access_count += 1
            
            # 접근 로그 기록
            self.access_log.append({
                "result_id": result_id,
                "requester_id": requester_id,
                "accessed_at": datetime.now(),
                "result_type": shared_result.result_type.value
            })
            
            self.stats["results_accessed"] += 1
            self.logger.debug(f"결과 조회: {requester_id} -> {result_id}")
            
            return shared_result.result
            
        except Exception as e:
            self.logger.error(f"결과 조회 중 오류: {e}")
            return None
    
    async def search_results(self, requester_id: str,
                           result_type: Optional[ResultType] = None,
                           keywords: Optional[List[str]] = None,
                           tags: Optional[List[str]] = None,
                           shared_by: Optional[str] = None,
                           date_range: Optional[tuple] = None) -> List[str]:
        """
        결과 검색
        
        Args:
            requester_id: 요청자 에이전트 ID
            result_type: 결과 타입 필터
            keywords: 키워드 필터
            tags: 태그 필터
            shared_by: 공유자 필터
            date_range: 날짜 범위 필터 (start_date, end_date)
            
        Returns:
            List[str]: 검색된 결과 ID 목록
        """
        try:
            result_ids = set()
            
            # 타입별 검색
            if result_type:
                result_ids = self.index.search_by_type(result_type)
            else:
                # 모든 결과 ID 수집
                for type_results in self.index.type_index.values():
                    result_ids.update(type_results)
            
            # 키워드 필터링
            if keywords:
                keyword_results = set()
                for keyword in keywords:
                    keyword_results.update(self.index.search_by_keyword(keyword))
                result_ids = result_ids.intersection(keyword_results)
            
            # 태그 필터링
            if tags:
                tag_results = set()
                for tag in tags:
                    tag_results.update(self.index.search_by_tag(tag))
                result_ids = result_ids.intersection(tag_results)
            
            # 공유자 필터링
            if shared_by:
                agent_results = self.index.search_by_agent(shared_by)
                result_ids = result_ids.intersection(agent_results)
            
            # 날짜 범위 필터링
            if date_range:
                start_date, end_date = date_range
                date_results = self.index.search_by_date_range(start_date, end_date)
                result_ids = result_ids.intersection(date_results)
            
            # 접근 권한 확인
            accessible_results = []
            requester_groups = self.agent_groups.get(requester_id, set())
            
            for result_id in result_ids:
                shared_result = self.shared_results.get(result_id)
                if shared_result and shared_result.can_access(requester_id, requester_groups):
                    accessible_results.append(result_id)
            
            self.logger.debug(f"결과 검색: {requester_id} -> {len(accessible_results)}개 결과")
            return accessible_results
            
        except Exception as e:
            self.logger.error(f"결과 검색 중 오류: {e}")
            return []
    
    def subscribe_to_results(self, agent_id: str, result_type: ResultType):
        """결과 타입 구독"""
        self.subscribers[result_type].add(agent_id)
        self.logger.debug(f"결과 구독: {agent_id} -> {result_type.value}")
    
    def subscribe_to_keyword(self, agent_id: str, keyword: str):
        """키워드 구독"""
        self.keyword_subscribers[keyword.lower()].add(agent_id)
        self.logger.debug(f"키워드 구독: {agent_id} -> {keyword}")
    
    def unsubscribe_from_results(self, agent_id: str, result_type: ResultType):
        """결과 타입 구독 해제"""
        self.subscribers[result_type].discard(agent_id)
        self.logger.debug(f"결과 구독 해제: {agent_id} -> {result_type.value}")
    
    def add_agent_to_group(self, agent_id: str, group_name: str):
        """에이전트를 그룹에 추가"""
        self.agent_groups[agent_id].add(group_name)
        self.logger.debug(f"그룹 추가: {agent_id} -> {group_name}")
    
    def remove_agent_from_group(self, agent_id: str, group_name: str):
        """에이전트를 그룹에서 제거"""
        self.agent_groups[agent_id].discard(group_name)
        self.logger.debug(f"그룹 제거: {agent_id} -> {group_name}")
    
    async def _notify_subscribers(self, shared_result: SharedResult):
        """구독자들에게 새 결과 알림"""
        # 타입 구독자들에게 알림
        type_subscribers = self.subscribers.get(shared_result.result_type, set())
        
        # 키워드 구독자들에게 알림
        keyword_subscribers = set()
        keywords = shared_result.metadata.get("keywords", [])
        for keyword in keywords:
            keyword_subscribers.update(self.keyword_subscribers.get(keyword.lower(), set()))
        
        all_subscribers = type_subscribers.union(keyword_subscribers)
        
        # 공유자는 제외
        all_subscribers.discard(shared_result.shared_by)
        
        # 알림 메시지 생성 및 전송 (실제 구현에서는 메시지 시스템 사용)
        for subscriber in all_subscribers:
            self.logger.debug(f"결과 알림: {subscriber} <- {shared_result.id}")
    
    async def cleanup_expired_results(self):
        """만료된 결과 정리"""
        expired_ids = []
        
        for result_id, shared_result in self.shared_results.items():
            if shared_result.is_expired():
                expired_ids.append(result_id)
        
        for result_id in expired_ids:
            shared_result = self.shared_results[result_id]
            self.index.remove_result(result_id, shared_result)
            del self.shared_results[result_id]
            self.stats["results_expired"] += 1
        
        if expired_ids:
            self.logger.info(f"만료된 결과 {len(expired_ids)}개 정리")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return {
            "total_results": len(self.shared_results),
            "active_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "keyword_subscribers": sum(len(subs) for subs in self.keyword_subscribers.values()),
            "agent_groups": len(self.agent_groups),
            "access_log_size": len(self.access_log),
            "stats": self.stats.copy()
        }


# 전역 결과 공유 시스템 인스턴스
_global_result_sharing: Optional[ResultSharingSystem] = None


def get_result_sharing_system() -> ResultSharingSystem:
    """전역 결과 공유 시스템 인스턴스 반환"""
    global _global_result_sharing
    if _global_result_sharing is None:
        _global_result_sharing = ResultSharingSystem()
    return _global_result_sharing