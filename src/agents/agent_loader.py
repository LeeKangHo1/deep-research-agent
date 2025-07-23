# src/agents/agent_loader.py

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import logging
import json
import yaml

from .base_agent import BaseAgent
from .agent_registry import AgentRegistry, get_agent_registry


class AgentLoader:
    """
    동적 에이전트 로딩 시스템
    
    파일 시스템에서 에이전트 클래스를 동적으로 로드하고 등록합니다.
    """
    
    def __init__(self, registry: Optional[AgentRegistry] = None):
        self.logger = logging.getLogger("agent_loader")
        self.registry = registry or get_agent_registry()
        
        # 로드된 모듈 추적
        self._loaded_modules: Dict[str, Any] = {}
        
        # 에이전트 설정 파일 경로
        self.config_paths = [
            "config/agents.json",
            "config/agents.yaml",
            "config/agents.yml"
        ]
        
        self.logger.info("에이전트 로더 초기화 완료")
    
    def load_agent_from_file(self, file_path: str, agent_name: Optional[str] = None) -> bool:
        """
        파일에서 에이전트 클래스를 로드
        
        Args:
            file_path: 에이전트 파일 경로
            agent_name: 특정 에이전트 이름 (None이면 모든 에이전트)
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"파일을 찾을 수 없음: {file_path}")
                return False
            
            # 모듈 이름 생성
            module_name = f"dynamic_agent_{file_path.stem}"
            
            # 모듈 스펙 생성
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                self.logger.error(f"모듈 스펙 생성 실패: {file_path}")
                return False
            
            # 모듈 로드
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # 에이전트 클래스 검색
            loaded_count = 0
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseAgent) and 
                    obj != BaseAgent and 
                    (agent_name is None or name == agent_name)):
                    
                    # 에이전트 타입 결정
                    agent_type = getattr(obj, 'AGENT_TYPE', name.lower())
                    
                    # 레지스트리에 등록
                    self.registry.register_agent_class(agent_type, obj)
                    loaded_count += 1
                    
                    self.logger.info(f"에이전트 로드 완료: {name} -> {agent_type}")
            
            if loaded_count > 0:
                self._loaded_modules[module_name] = module
                return True
            else:
                self.logger.warning(f"에이전트 클래스를 찾을 수 없음: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"에이전트 로드 중 오류: {e}")
            return False
    
    def load_agents_from_directory(self, directory: str, recursive: bool = True) -> int:
        """
        디렉토리에서 모든 에이전트 파일을 로드
        
        Args:
            directory: 검색할 디렉토리
            recursive: 하위 디렉토리 포함 여부
            
        Returns:
            int: 로드된 에이전트 수
        """
        try:
            directory = Path(directory)
            if not directory.exists():
                self.logger.error(f"디렉토리를 찾을 수 없음: {directory}")
                return 0
            
            loaded_count = 0
            pattern = "**/*.py" if recursive else "*.py"
            
            for file_path in directory.glob(pattern):
                if file_path.name.startswith("__"):
                    continue
                
                if self.load_agent_from_file(str(file_path)):
                    loaded_count += 1
            
            self.logger.info(f"디렉토리에서 {loaded_count}개 에이전트 로드: {directory}")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"디렉토리 로드 중 오류: {e}")
            return 0
    
    def load_agent_from_module(self, module_path: str, agent_name: Optional[str] = None) -> bool:
        """
        모듈 경로에서 에이전트 로드
        
        Args:
            module_path: 모듈 경로 (예: "agents.research_agent")
            agent_name: 특정 에이전트 이름
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 모듈 임포트
            module = importlib.import_module(module_path)
            
            # 에이전트 클래스 검색
            loaded_count = 0
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseAgent) and 
                    obj != BaseAgent and 
                    (agent_name is None or name == agent_name)):
                    
                    # 에이전트 타입 결정
                    agent_type = getattr(obj, 'AGENT_TYPE', name.lower())
                    
                    # 레지스트리에 등록
                    self.registry.register_agent_class(agent_type, obj)
                    loaded_count += 1
                    
                    self.logger.info(f"모듈에서 에이전트 로드: {name} -> {agent_type}")
            
            if loaded_count > 0:
                self._loaded_modules[module_path] = module
                return True
            else:
                self.logger.warning(f"에이전트 클래스를 찾을 수 없음: {module_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"모듈 로드 중 오류: {e}")
            return False
    
    def load_from_config(self, config_path: Optional[str] = None) -> int:
        """
        설정 파일에서 에이전트 로드
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 경로 검색)
            
        Returns:
            int: 로드된 에이전트 수
        """
        config_file = None
        
        if config_path:
            config_file = Path(config_path)
        else:
            # 기본 설정 파일 검색
            for path in self.config_paths:
                if Path(path).exists():
                    config_file = Path(path)
                    break
        
        if not config_file or not config_file.exists():
            self.logger.warning("에이전트 설정 파일을 찾을 수 없음")
            return 0
        
        try:
            # 설정 파일 로드
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            loaded_count = 0
            agents_config = config.get('agents', [])
            
            for agent_config in agents_config:
                agent_type = agent_config.get('type')
                module_path = agent_config.get('module')
                file_path = agent_config.get('file')
                class_name = agent_config.get('class')
                settings = agent_config.get('config', {})
                
                success = False
                
                if file_path:
                    success = self.load_agent_from_file(file_path, class_name)
                elif module_path:
                    success = self.load_agent_from_module(module_path, class_name)
                
                if success:
                    loaded_count += 1
                    
                    # 설정 적용
                    if agent_type and settings:
                        self.registry._agent_configs[agent_type] = settings
            
            self.logger.info(f"설정 파일에서 {loaded_count}개 에이전트 로드: {config_file}")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류: {e}")
            return 0
    
    def reload_agent(self, module_name: str) -> bool:
        """
        에이전트 모듈 재로드
        
        Args:
            module_name: 재로드할 모듈 이름
            
        Returns:
            bool: 재로드 성공 여부
        """
        try:
            if module_name in self._loaded_modules:
                module = self._loaded_modules[module_name]
                importlib.reload(module)
                
                # 에이전트 클래스 재등록
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseAgent) and obj != BaseAgent:
                        agent_type = getattr(obj, 'AGENT_TYPE', name.lower())
                        self.registry.register_agent_class(agent_type, obj)
                
                self.logger.info(f"에이전트 모듈 재로드 완료: {module_name}")
                return True
            else:
                self.logger.warning(f"로드되지 않은 모듈: {module_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"모듈 재로드 중 오류: {e}")
            return False
    
    def unload_agent(self, module_name: str) -> bool:
        """
        에이전트 모듈 언로드
        
        Args:
            module_name: 언로드할 모듈 이름
            
        Returns:
            bool: 언로드 성공 여부
        """
        try:
            if module_name in self._loaded_modules:
                # 시스템 모듈에서 제거
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # 로드된 모듈 목록에서 제거
                del self._loaded_modules[module_name]
                
                self.logger.info(f"에이전트 모듈 언로드 완료: {module_name}")
                return True
            else:
                self.logger.warning(f"로드되지 않은 모듈: {module_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"모듈 언로드 중 오류: {e}")
            return False
    
    def get_loaded_modules(self) -> List[str]:
        """로드된 모듈 목록 반환"""
        return list(self._loaded_modules.keys())
    
    def auto_discover_agents(self, base_path: str = "src/agents") -> int:
        """
        자동으로 에이전트 발견 및 로드
        
        Args:
            base_path: 검색 시작 경로
            
        Returns:
            int: 발견된 에이전트 수
        """
        try:
            base_path = Path(base_path)
            if not base_path.exists():
                self.logger.warning(f"기본 경로가 존재하지 않음: {base_path}")
                return 0
            
            discovered_count = 0
            
            # Python 파일에서 에이전트 검색
            for py_file in base_path.rglob("*.py"):
                if py_file.name.startswith("__") or py_file.name == "base_agent.py":
                    continue
                
                try:
                    # 파일 내용 검사하여 BaseAgent 상속 클래스 확인
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "BaseAgent" in content and "class " in content:
                            if self.load_agent_from_file(str(py_file)):
                                discovered_count += 1
                except Exception as e:
                    self.logger.debug(f"파일 검사 중 오류 ({py_file}): {e}")
            
            self.logger.info(f"자동 발견으로 {discovered_count}개 에이전트 로드")
            return discovered_count
            
        except Exception as e:
            self.logger.error(f"자동 발견 중 오류: {e}")
            return 0


# 전역 에이전트 로더 인스턴스
_global_loader: Optional[AgentLoader] = None


def get_agent_loader() -> AgentLoader:
    """전역 에이전트 로더 인스턴스 반환"""
    global _global_loader
    if _global_loader is None:
        _global_loader = AgentLoader()
    return _global_loader