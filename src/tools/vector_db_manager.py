# src/tools/vector_db_manager.py

"""
벡터 데이터베이스 관리 도구 구현
VectorDBManager 클래스 구현
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from langchain_openai import OpenAIEmbeddings

from src.models.search import SearchResult
from src.models.vector_db import VectorDBEntry


class VectorDBManager:
    """
    벡터 데이터베이스 관리 도구
    
    검색 결과를 벡터화하여 저장하고, 의미적 유사성 기반 검색을 제공합니다.
    데이터 신선도를 평가하고 오래된 데이터를 업데이트하는 기능을 제공합니다.
    
    Attributes:
        embedding_model_name (str): 임베딩 모델 이름
        persist_directory (str): 벡터 DB 저장 디렉토리
        collection_name (str): 컬렉션 이름
        client (chromadb.Client): ChromaDB 클라이언트
        collection (chromadb.Collection): ChromaDB 컬렉션
        embedding_function: 임베딩 함수
        openai_api_key (str): OpenAI API 키
    """
    
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-ada-002",
        persist_directory: str = "./chroma_db",
        collection_name: str = "search_results",
    ):
        """
        VectorDBManager 초기화
        
        Args:
            embedding_model_name (str, optional): 임베딩 모델 이름. 기본값은 "text-embedding-ada-002".
            persist_directory (str, optional): 벡터 DB 저장 디렉토리. 기본값은 "./chroma_db".
            collection_name (str, optional): 컬렉션 이름. 기본값은 "search_results".
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # OpenAI API 키 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logging.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 임베딩 함수 초기화
        self.embedding_function = None
        
        # ChromaDB 클라이언트 및 컬렉션 초기화
        self.client = None
        self.collection = None
        
        # 초기화
        self._initialize_vector_db()
        
    def _initialize_embedding_function(self) -> None:
        """
        임베딩 함수 초기화
        
        OpenAI 임베딩 모델을 사용하여 텍스트를 벡터로 변환하는 함수를 초기화합니다.
        API 키가 없는 경우 기본 ChromaDB 임베딩 함수를 사용합니다.
        
        Raises:
            ValueError: 임베딩 모델 초기화 실패 시 발생
        """
        try:
            # OpenAI API 키가 있는 경우 OpenAI 임베딩 사용
            if self.openai_api_key:
                logging.info(f"OpenAI 임베딩 모델 '{self.embedding_model_name}' 초기화 중...")
                
                # LangChain OpenAI 임베딩 초기화
                openai_embeddings = OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    openai_api_key=self.openai_api_key
                )
                
                # ChromaDB와 호환되는 임베딩 함수 생성
                self.embedding_function = embedding_functions.LangchainEmbeddingFunction(
                    model_name=self.embedding_model_name,
                    langchain_embeddings=openai_embeddings
                )
                
                logging.info("OpenAI 임베딩 모델 초기화 완료")
            else:
                # API 키가 없는 경우 기본 ChromaDB 임베딩 함수 사용
                logging.warning("OpenAI API 키가 없어 기본 ChromaDB 임베딩 함수를 사용합니다.")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                
        except Exception as e:
            error_msg = f"임베딩 모델 초기화 실패: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_vector_db(self) -> None:
        """
        벡터 데이터베이스 초기화
        
        ChromaDB 클라이언트를 초기화하고 컬렉션을 생성합니다.
        
        Raises:
            RuntimeError: 벡터 DB 초기화 실패 시 발생
        """
        try:
            logging.info(f"벡터 DB 초기화 중... (저장 디렉토리: {self.persist_directory})")
            
            # 임베딩 함수 초기화
            if not self.embedding_function:
                self._initialize_embedding_function()
            
            # 저장 디렉토리가 없으면 생성
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # ChromaDB 클라이언트 초기화
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 가져오기 또는 생성
            try:
                # 기존 컬렉션 가져오기 시도
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logging.info(f"기존 컬렉션 '{self.collection_name}'을 불러왔습니다.")
            except Exception:
                # 컬렉션이 없으면 새로 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "검색 결과 저장소"}
                )
                logging.info(f"새 컬렉션 '{self.collection_name}'을 생성했습니다.")
                
            logging.info("벡터 DB 초기화 완료")
            
        except Exception as e:
            error_msg = f"벡터 DB 초기화 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def _get_embedding(self, text: str) -> List[float]:
        """
        텍스트의 임베딩 벡터 생성
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
            
        Raises:
            ValueError: 임베딩 생성 실패 시 발생
        """
        try:
            if not self.embedding_function:
                self._initialize_embedding_function()
                
            # 임베딩 함수를 사용하여 텍스트를 벡터로 변환
            embeddings = self.embedding_function([text])
            
            # 단일 임베딩 반환
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                raise ValueError("임베딩 생성 결과가 비어 있습니다.")
                
        except Exception as e:
            error_msg = f"임베딩 생성 실패: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트의 임베딩 벡터 생성
        
        Args:
            texts (List[str]): 임베딩할 텍스트 목록
            
        Returns:
            List[List[float]]: 임베딩 벡터 목록
            
        Raises:
            ValueError: 임베딩 생성 실패 시 발생
        """
        try:
            if not texts:
                return []
                
            if not self.embedding_function:
                self._initialize_embedding_function()
                
            # 임베딩 함수를 사용하여 텍스트 목록을 벡터로 변환
            embeddings = self.embedding_function(texts)
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("임베딩 생성 결과가 비어 있습니다.")
                
            return embeddings
            
        except Exception as e:
            error_msg = f"임베딩 생성 실패: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    def preprocess_text_for_embedding(self, text: str, max_length: int = 8000) -> str:
        """
        임베딩을 위한 텍스트 전처리
        
        텍스트를 정리하고 임베딩에 적합한 형태로 변환합니다.
        
        Args:
            text (str): 전처리할 텍스트
            max_length (int, optional): 최대 텍스트 길이. 기본값은 8000.
            
        Returns:
            str: 전처리된 텍스트
        """
        if not text:
            return ""
            
        # 텍스트 정리
        processed_text = text.strip()
        
        # 여러 줄바꿈 제거
        processed_text = ' '.join([line.strip() for line in processed_text.splitlines() if line.strip()])
        
        # 여러 공백 제거
        import re
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # 최대 길이 제한
        if len(processed_text) > max_length:
            processed_text = processed_text[:max_length]
            
        return processed_text
            
    def reconnect(self) -> bool:
        """
        벡터 DB에 재연결
        
        연결이 끊어진 경우 다시 연결을 시도합니다.
        
        Returns:
            bool: 재연결 성공 여부
        """
        try:
            # 기존 연결 정리
            self.client = None
            self.collection = None
            
            # 다시 초기화
            self._initialize_vector_db()
            return True
        except Exception as e:
            logging.error(f"벡터 DB 재연결 실패: {str(e)}")
            return False
            
    def create_index(self, index_name: str = None) -> bool:
        """
        인덱스 생성
        
        효율적인 검색을 위한 인덱스를 생성합니다.
        ChromaDB는 내부적으로 인덱스를 관리하므로, 이 메서드는 주로 인덱스 최적화를 위한 것입니다.
        
        Args:
            index_name (str, optional): 인덱스 이름. 기본값은 None으로, 컬렉션 이름을 사용합니다.
            
        Returns:
            bool: 인덱스 생성 성공 여부
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 인덱스 이름이 지정되지 않은 경우 컬렉션 이름 사용
            actual_index_name = index_name or f"{self.collection_name}_index"
            
            logging.info(f"인덱스 '{actual_index_name}' 생성 중...")
            
            # ChromaDB는 내부적으로 인덱스를 관리하므로 별도의 인덱스 생성이 필요 없음
            # 대신 컬렉션 메타데이터에 인덱스 정보 추가
            collection_metadata = self.collection.metadata or {}
            collection_metadata["index_name"] = actual_index_name
            collection_metadata["index_created_at"] = datetime.now().isoformat()
            
            # 컬렉션 메타데이터 업데이트
            # 참고: ChromaDB API에서는 컬렉션 메타데이터를 직접 업데이트할 수 없음
            # 따라서 이 정보는 내부적으로만 유지됨
            
            logging.info(f"인덱스 '{actual_index_name}' 생성 완료")
            return True
            
        except Exception as e:
            error_msg = f"인덱스 생성 실패: {str(e)}"
            logging.error(error_msg)
            return False
            
    def optimize_index(self) -> bool:
        """
        인덱스 최적화
        
        검색 성능 향상을 위해 인덱스를 최적화합니다.
        
        Returns:
            bool: 최적화 성공 여부
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            logging.info("인덱스 최적화 중...")
            
            # ChromaDB는 내부적으로 인덱스를 관리하므로 별도의 최적화 작업이 필요 없음
            # 대신 컬렉션 메타데이터에 최적화 정보 추가
            collection_metadata = self.collection.metadata or {}
            collection_metadata["last_optimized_at"] = datetime.now().isoformat()
            
            logging.info("인덱스 최적화 완료")
            return True
            
        except Exception as e:
            error_msg = f"인덱스 최적화 실패: {str(e)}"
            logging.error(error_msg)
            return False
            
    def get_index_stats(self) -> Dict[str, Any]:
        """
        인덱스 통계 정보 조회
        
        인덱스의 크기, 항목 수 등 통계 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 인덱스 통계 정보
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 컬렉션 항목 수 조회
            count = self.collection.count()
            
            # 컬렉션 메타데이터 조회
            collection_metadata = self.collection.metadata or {}
            
            # 통계 정보 구성
            stats = {
                "count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "last_updated": datetime.now().isoformat()
            }
            
            # 컬렉션 메타데이터에서 추가 정보 가져오기
            if "index_name" in collection_metadata:
                stats["index_name"] = collection_metadata["index_name"]
            if "index_created_at" in collection_metadata:
                stats["index_created_at"] = collection_metadata["index_created_at"]
            if "last_optimized_at" in collection_metadata:
                stats["last_optimized_at"] = collection_metadata["last_optimized_at"]
                
            return stats
            
        except Exception as e:
            error_msg = f"인덱스 통계 정보 조회 실패: {str(e)}"
            logging.error(error_msg)
            return {"error": str(e)}
            
    def reset_index(self, confirm: bool = False) -> bool:
        """
        인덱스 초기화
        
        모든 데이터를 삭제하고 인덱스를 초기화합니다.
        
        Args:
            confirm (bool, optional): 초기화 확인. 기본값은 False.
            
        Returns:
            bool: 초기화 성공 여부
        """
        if not confirm:
            logging.warning("인덱스 초기화가 확인되지 않았습니다. confirm=True로 호출하세요.")
            return False
            
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            logging.info(f"컬렉션 '{self.collection_name}' 초기화 중...")
            
            # 컬렉션 삭제
            self.client.delete_collection(self.collection_name)
            
            # 컬렉션 다시 생성
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "검색 결과 저장소"}
            )
            
            logging.info(f"컬렉션 '{self.collection_name}' 초기화 완료")
            return True
            
        except Exception as e:
            error_msg = f"인덱스 초기화 실패: {str(e)}"
            logging.error(error_msg)
            return False    
    def prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        메타데이터 준비
        
        ChromaDB에 저장하기 위한 메타데이터를 준비합니다.
        
        Args:
            metadata (Dict[str, Any]): 원본 메타데이터
            
        Returns:
            Dict[str, Any]: 처리된 메타데이터
        """
        if not metadata:
            return {}
            
        # 메타데이터 복사
        processed_metadata = metadata.copy()
        
        # 타임스탬프 추가
        if "timestamp" not in processed_metadata:
            processed_metadata["timestamp"] = datetime.now().isoformat()
            
        # 중첩된 객체나 리스트를 문자열로 변환 (ChromaDB 요구사항)
        for key, value in processed_metadata.items():
            if isinstance(value, (dict, list)):
                processed_metadata[key] = str(value)
            elif isinstance(value, datetime):
                processed_metadata[key] = value.isoformat()
                
        return processed_metadata
        
    def extract_metadata_from_search_result(self, result: SearchResult) -> Dict[str, Any]:
        """
        SearchResult에서 메타데이터 추출
        
        Args:
            result (SearchResult): 검색 결과 객체
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        metadata = {
            "title": result.title,
            "url": str(result.url),
            "score": result.score,
            "source_type": result.source_type or "web",
        }
        
        # 발행 날짜가 있으면 추가
        if result.published_date:
            metadata["published_date"] = result.published_date.isoformat()
            
        # 메타데이터가 있으면 병합
        if result.metadata:
            for key, value in result.metadata.items():
                if key not in metadata:  # 기존 키를 덮어쓰지 않음
                    metadata[key] = value
                    
        return metadata
        
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        항목의 메타데이터 업데이트
        
        Args:
            id (str): 항목 ID
            metadata (Dict[str, Any]): 새 메타데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 항목 조회
            result = self.collection.get(ids=[id])
            
            if not result or not result["ids"]:
                logging.warning(f"ID '{id}'에 해당하는 항목을 찾을 수 없습니다.")
                return False
                
            # 기존 메타데이터 가져오기
            existing_metadata = result["metadatas"][0] if result["metadatas"] else {}
            
            # 새 메타데이터 준비
            updated_metadata = {**existing_metadata, **self.prepare_metadata(metadata)}
            
            # 메타데이터 업데이트
            self.collection.update(
                ids=[id],
                metadatas=[updated_metadata]
            )
            
            logging.info(f"ID '{id}'의 메타데이터 업데이트 완료")
            return True
            
        except Exception as e:
            error_msg = f"메타데이터 업데이트 실패: {str(e)}"
            logging.error(error_msg)
            return False
            
    def filter_by_metadata(self, query: Dict[str, Any]) -> List[str]:
        """
        메타데이터 기반 필터링
        
        Args:
            query (Dict[str, Any]): 메타데이터 쿼리
            
        Returns:
            List[str]: 필터링된 항목 ID 목록
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # ChromaDB의 where 필터를 사용하여 메타데이터 기반 검색
            result = self.collection.get(where=query)
            
            if not result or not result["ids"]:
                return []
                
            return result["ids"]
            
        except Exception as e:
            error_msg = f"메타데이터 필터링 실패: {str(e)}"
            logging.error(error_msg)
            return [] 
    def store_search_result(self, result: SearchResult, query: str = None) -> str:
        """
        검색 결과를 벡터화하여 저장
        
        Args:
            result (SearchResult): 저장할 검색 결과
            query (str, optional): 검색 쿼리. 기본값은 None.
            
        Returns:
            str: 저장된 항목의 ID
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 검색 결과 내용 전처리
            content = self.preprocess_text_for_embedding(result.content)
            
            # 메타데이터 준비
            metadata = self.extract_metadata_from_search_result(result)
            
            # 검색 쿼리가 있으면 메타데이터에 추가
            if query:
                metadata["query"] = query
                
            # 메타데이터 처리
            processed_metadata = self.prepare_metadata(metadata)
            
            # 임베딩 생성
            embedding = self._get_embedding(content)
            
            # 고유 ID 생성
            id = str(uuid.uuid4())
            
            # ChromaDB에 저장
            self.collection.add(
                ids=[id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[processed_metadata]
            )
            
            logging.info(f"검색 결과 저장 완료 (ID: {id})")
            return id
            
        except Exception as e:
            error_msg = f"검색 결과 저장 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def store_multiple_search_results(self, results: List[SearchResult], query: str = None) -> List[str]:
        """
        여러 검색 결과를 벡터화하여 저장
        
        Args:
            results (List[SearchResult]): 저장할 검색 결과 목록
            query (str, optional): 검색 쿼리. 기본값은 None.
            
        Returns:
            List[str]: 저장된 항목의 ID 목록
        """
        try:
            if not results:
                return []
                
            if not self.collection:
                self._initialize_vector_db()
                
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            # 각 검색 결과 처리
            for result in results:
                # 검색 결과 내용 전처리
                content = self.preprocess_text_for_embedding(result.content)
                
                # 메타데이터 준비
                metadata = self.extract_metadata_from_search_result(result)
                
                # 검색 쿼리가 있으면 메타데이터에 추가
                if query:
                    metadata["query"] = query
                    
                # 메타데이터 처리
                processed_metadata = self.prepare_metadata(metadata)
                
                # 고유 ID 생성
                id = str(uuid.uuid4())
                
                ids.append(id)
                documents.append(content)
                metadatas.append(processed_metadata)
                
            # 임베딩 일괄 생성
            embeddings = self.create_embeddings(documents)
            
            # ChromaDB에 일괄 저장
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logging.info(f"{len(ids)}개의 검색 결과 저장 완료")
            return ids
            
        except Exception as e:
            error_msg = f"검색 결과 일괄 저장 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def store_vector_db_entry(self, entry: VectorDBEntry) -> str:
        """
        VectorDBEntry 객체 저장
        
        Args:
            entry (VectorDBEntry): 저장할 VectorDBEntry 객체
            
        Returns:
            str: 저장된 항목의 ID
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 메타데이터 처리
            processed_metadata = self.prepare_metadata(entry.metadata)
            
            # 벡터가 없으면 생성
            embedding = entry.vector
            if not embedding or len(embedding) == 0:
                embedding = self._get_embedding(entry.content)
                
            # ChromaDB에 저장
            self.collection.add(
                ids=[entry.id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[processed_metadata]
            )
            
            logging.info(f"VectorDBEntry 저장 완료 (ID: {entry.id})")
            return entry.id
            
        except Exception as e:
            error_msg = f"VectorDBEntry 저장 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def store_content_with_metadata(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        id: str = None
    ) -> str:
        """
        콘텐츠와 메타데이터 저장
        
        Args:
            content (str): 저장할 콘텐츠
            metadata (Dict[str, Any]): 저장할 메타데이터
            id (str, optional): 항목 ID. 기본값은 None으로, 자동 생성됩니다.
            
        Returns:
            str: 저장된 항목의 ID
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 콘텐츠 전처리
            processed_content = self.preprocess_text_for_embedding(content)
            
            # 메타데이터 처리
            processed_metadata = self.prepare_metadata(metadata)
            
            # 임베딩 생성
            embedding = self._get_embedding(processed_content)
            
            # ID가 없으면 생성
            if not id:
                id = str(uuid.uuid4())
                
            # ChromaDB에 저장
            self.collection.add(
                ids=[id],
                embeddings=[embedding],
                documents=[processed_content],
                metadatas=[processed_metadata]
            )
            
            logging.info(f"콘텐츠 저장 완료 (ID: {id})")
            return id
            
        except Exception as e:
            error_msg = f"콘텐츠 저장 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def store_batch_with_metadata(
        self, 
        contents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str] = None
    ) -> List[str]:
        """
        여러 콘텐츠와 메타데이터 일괄 저장
        
        Args:
            contents (List[str]): 저장할 콘텐츠 목록
            metadatas (List[Dict[str, Any]]): 저장할 메타데이터 목록
            ids (List[str], optional): 항목 ID 목록. 기본값은 None으로, 자동 생성됩니다.
            
        Returns:
            List[str]: 저장된 항목의 ID 목록
        """
        try:
            if not contents or not metadatas:
                return []
                
            if len(contents) != len(metadatas):
                raise ValueError("콘텐츠와 메타데이터 목록의 길이가 일치해야 합니다.")
                
            if not self.collection:
                self._initialize_vector_db()
                
            # ID가 없으면 생성
            if not ids:
                ids = [str(uuid.uuid4()) for _ in range(len(contents))]
            elif len(ids) != len(contents):
                raise ValueError("ID 목록의 길이가 콘텐츠 목록의 길이와 일치해야 합니다.")
                
            # 콘텐츠 전처리
            processed_contents = [self.preprocess_text_for_embedding(content) for content in contents]
            
            # 메타데이터 처리
            processed_metadatas = [self.prepare_metadata(metadata) for metadata in metadatas]
            
            # 임베딩 일괄 생성
            embeddings = self.create_embeddings(processed_contents)
            
            # ChromaDB에 일괄 저장
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=processed_contents,
                metadatas=processed_metadatas
            )
            
            logging.info(f"{len(ids)}개의 콘텐츠 일괄 저장 완료")
            return ids
            
        except Exception as e:
            error_msg = f"콘텐츠 일괄 저장 실패: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
    def get_metadata(self, id: str) -> Dict[str, Any]:
        """
        항목의 메타데이터 조회
        
        Args:
            id (str): 항목 ID
            
        Returns:
            Dict[str, Any]: 메타데이터
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 항목 조회
            result = self.collection.get(ids=[id])
            
            if not result or not result["ids"]:
                logging.warning(f"ID '{id}'에 해당하는 항목을 찾을 수 없습니다.")
                return {}
                
            # 메타데이터 반환
            return result["metadatas"][0] if result["metadatas"] else {}
            
        except Exception as e:
            error_msg = f"메타데이터 조회 실패: {str(e)}"
            logging.error(error_msg)
            return {}
            
    def search_by_text(
        self, 
        query_text: str, 
        top_k: int = 5, 
        threshold: float = 0.0,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        텍스트 기반 의미적 유사성 검색
        
        Args:
            query_text (str): 검색 쿼리 텍스트
            top_k (int, optional): 반환할 최대 결과 수. 기본값은 5.
            threshold (float, optional): 유사도 임계값. 기본값은 0.0.
            metadata_filter (Dict[str, Any], optional): 메타데이터 필터. 기본값은 None.
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 쿼리 텍스트 전처리
            processed_query = self.preprocess_text_for_embedding(query_text)
            
            # 쿼리 임베딩 생성
            query_embedding = self._get_embedding(processed_query)
            
            # ChromaDB 검색 수행
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            
            if not results or not results["ids"] or not results["ids"][0]:
                return []
                
            # 결과 처리
            search_results = []
            for i in range(len(results["ids"][0])):
                # 유사도 점수 계산 (ChromaDB는 거리를 반환하므로 유사도로 변환)
                # 거리가 작을수록 유사도가 높음
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                similarity = 1.0 - min(distance, 1.0)  # 거리를 유사도로 변환 (0~1 범위)
                
                # 임계값 필터링
                if similarity < threshold:
                    continue
                    
                # 결과 구성
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity
                }
                
                search_results.append(result)
                
            return search_results
            
        except Exception as e:
            error_msg = f"텍스트 기반 검색 실패: {str(e)}"
            logging.error(error_msg)
            return []
            
    def search_by_vector(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        threshold: float = 0.0,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        벡터 기반 의미적 유사성 검색
        
        Args:
            query_vector (List[float]): 검색 쿼리 벡터
            top_k (int, optional): 반환할 최대 결과 수. 기본값은 5.
            threshold (float, optional): 유사도 임계값. 기본값은 0.0.
            metadata_filter (Dict[str, Any], optional): 메타데이터 필터. 기본값은 None.
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # ChromaDB 검색 수행
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=metadata_filter
            )
            
            if not results or not results["ids"] or not results["ids"][0]:
                return []
                
            # 결과 처리
            search_results = []
            for i in range(len(results["ids"][0])):
                # 유사도 점수 계산
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                similarity = 1.0 - min(distance, 1.0)
                
                # 임계값 필터링
                if similarity < threshold:
                    continue
                    
                # 결과 구성
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity
                }
                
                search_results.append(result)
                
            return search_results
            
        except Exception as e:
            error_msg = f"벡터 기반 검색 실패: {str(e)}"
            logging.error(error_msg)
            return []
            
    def search_similar_to_id(
        self, 
        id: str, 
        top_k: int = 5, 
        threshold: float = 0.0,
        metadata_filter: Dict[str, Any] = None,
        include_self: bool = False
    ) -> List[Dict[str, Any]]:
        """
        ID 기반 유사 항목 검색
        
        Args:
            id (str): 기준 항목 ID
            top_k (int, optional): 반환할 최대 결과 수. 기본값은 5.
            threshold (float, optional): 유사도 임계값. 기본값은 0.0.
            metadata_filter (Dict[str, Any], optional): 메타데이터 필터. 기본값은 None.
            include_self (bool, optional): 자기 자신을 결과에 포함할지 여부. 기본값은 False.
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 항목 조회
            result = self.collection.get(ids=[id])
            
            if not result or not result["ids"]:
                logging.warning(f"ID '{id}'에 해당하는 항목을 찾을 수 없습니다.")
                return []
                
            # 항목의 임베딩 벡터 가져오기
            if "embeddings" not in result or not result["embeddings"]:
                logging.warning(f"ID '{id}'에 해당하는 항목의 임베딩을 찾을 수 없습니다.")
                return []
                
            query_vector = result["embeddings"][0]
            
            # 벡터 기반 검색 수행
            results = self.search_by_vector(
                query_vector=query_vector,
                top_k=top_k + (0 if include_self else 1),  # 자기 자신을 제외하려면 하나 더 요청
                threshold=threshold,
                metadata_filter=metadata_filter
            )
            
            # 자기 자신 제외
            if not include_self:
                results = [r for r in results if r["id"] != id]
                
            # 최대 결과 수 제한
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"ID 기반 유사 항목 검색 실패: {str(e)}"
            logging.error(error_msg)
            return []
            
    def calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        
        Args:
            vector1 (List[float]): 첫 번째 벡터
            vector2 (List[float]): 두 번째 벡터
            
        Returns:
            float: 코사인 유사도 (0.0 ~ 1.0)
        """
        try:
            import numpy as np
            
            # 벡터 검증
            if not vector1 or not vector2:
                return 0.0
                
            if len(vector1) != len(vector2):
                logging.warning(f"벡터 길이가 일치하지 않습니다: {len(vector1)} vs {len(vector2)}")
                return 0.0
                
            # NumPy 배열로 변환
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 벡터 크기 계산
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            # 0으로 나누기 방지
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # 코사인 유사도 계산
            cos_sim = np.dot(v1, v2) / (norm1 * norm2)
            
            # 범위를 0.0 ~ 1.0으로 제한
            return max(0.0, min(1.0, cos_sim))
            
        except Exception as e:
            logging.error(f"유사도 계산 실패: {str(e)}")
            return 0.0
            
    def filter_by_similarity(
        self, 
        results: List[Dict[str, Any]], 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        유사도 임계값 기반 필터링
        
        Args:
            results (List[Dict[str, Any]]): 필터링할 검색 결과 목록
            threshold (float, optional): 유사도 임계값. 기본값은 0.7.
            
        Returns:
            List[Dict[str, Any]]: 필터링된 검색 결과 목록
        """
        if not results:
            return []
            
        # 임계값 기반 필터링
        filtered_results = [r for r in results if r.get("similarity", 0.0) >= threshold]
        
        return filtered_results
        
    def rank_by_similarity(
        self, 
        results: List[Dict[str, Any]], 
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        유사도 기반 결과 정렬
        
        Args:
            results (List[Dict[str, Any]]): 정렬할 검색 결과 목록
            reverse (bool, optional): 내림차순 정렬 여부. 기본값은 True.
            
        Returns:
            List[Dict[str, Any]]: 정렬된 검색 결과 목록
        """
        if not results:
            return []
            
        # 유사도 기반 정렬
        sorted_results = sorted(
            results, 
            key=lambda x: x.get("similarity", 0.0), 
            reverse=reverse
        )
        
        return sorted_results
        
    def combine_similarity_with_metadata(
        self, 
        results: List[Dict[str, Any]], 
        metadata_key: str, 
        weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        유사도와 메타데이터 점수를 결합하여 최종 점수 계산
        
        Args:
            results (List[Dict[str, Any]]): 검색 결과 목록
            metadata_key (str): 메타데이터 점수 키
            weight (float, optional): 유사도 가중치. 기본값은 0.5.
            
        Returns:
            List[Dict[str, Any]]: 최종 점수가 추가된 검색 결과 목록
        """
        if not results:
            return []
            
        combined_results = []
        
        for result in results:
            # 유사도 점수
            similarity = result.get("similarity", 0.0)
            
            # 메타데이터 점수
            metadata = result.get("metadata", {})
            metadata_score = float(metadata.get(metadata_key, 0.0))
            
            # 최종 점수 계산
            final_score = (weight * similarity) + ((1 - weight) * metadata_score)
            
            # 결과에 최종 점수 추가
            result_with_score = result.copy()
            result_with_score["final_score"] = final_score
            
            combined_results.append(result_with_score)
            
        # 최종 점수 기준으로 정렬
        return sorted(combined_results, key=lambda x: x["final_score"], reverse=True)    
    def calculate_freshness_score(
        self, 
        timestamp: str, 
        max_age_days: int = 30
    ) -> float:
        """
        타임스탬프 기반 신선도 점수 계산
        
        Args:
            timestamp (str): ISO 형식 타임스탬프
            max_age_days (int, optional): 최대 나이(일). 기본값은 30.
            
        Returns:
            float: 신선도 점수 (0.0 ~ 1.0)
        """
        try:
            # 타임스탬프 파싱
            if not timestamp:
                return 0.0
                
            timestamp_dt = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            
            # 경과 시간 계산 (일)
            age_days = (current_time - timestamp_dt).total_seconds() / (24 * 60 * 60)
            
            # 신선도 점수 계산 (선형 감소)
            if age_days <= 0:
                return 1.0
            elif age_days >= max_age_days:
                return 0.0
            else:
                return 1.0 - (age_days / max_age_days)
                
        except Exception as e:
            logging.error(f"신선도 점수 계산 실패: {str(e)}")
            return 0.0
            
    def evaluate_data_freshness(
        self, 
        results: List[Dict[str, Any]], 
        max_age_days: int = 30,
        timestamp_key: str = "timestamp"
    ) -> List[Dict[str, Any]]:
        """
        검색 결과의 데이터 신선도 평가
        
        Args:
            results (List[Dict[str, Any]]): 검색 결과 목록
            max_age_days (int, optional): 최대 나이(일). 기본값은 30.
            timestamp_key (str, optional): 타임스탬프 메타데이터 키. 기본값은 "timestamp".
            
        Returns:
            List[Dict[str, Any]]: 신선도 점수가 추가된 검색 결과 목록
        """
        if not results:
            return []
            
        results_with_freshness = []
        
        for result in results:
            # 메타데이터에서 타임스탬프 가져오기
            metadata = result.get("metadata", {})
            timestamp = metadata.get(timestamp_key)
            
            # 신선도 점수 계산
            freshness_score = self.calculate_freshness_score(timestamp, max_age_days)
            
            # 결과에 신선도 점수 추가
            result_with_freshness = result.copy()
            result_with_freshness["freshness_score"] = freshness_score
            
            # 메타데이터에도 신선도 점수 추가
            metadata_with_freshness = metadata.copy()
            metadata_with_freshness["freshness_score"] = freshness_score
            result_with_freshness["metadata"] = metadata_with_freshness
            
            results_with_freshness.append(result_with_freshness)
            
        return results_with_freshness
        
    def filter_by_freshness(
        self, 
        results: List[Dict[str, Any]], 
        min_freshness: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        신선도 임계값 기반 필터링
        
        Args:
            results (List[Dict[str, Any]]): 필터링할 검색 결과 목록
            min_freshness (float, optional): 최소 신선도 점수. 기본값은 0.5.
            
        Returns:
            List[Dict[str, Any]]: 필터링된 검색 결과 목록
        """
        if not results:
            return []
            
        # 신선도 점수가 없는 경우 계산
        if "freshness_score" not in results[0]:
            results = self.evaluate_data_freshness(results)
            
        # 임계값 기반 필터링
        filtered_results = [r for r in results if r.get("freshness_score", 0.0) >= min_freshness]
        
        return filtered_results
        
    def combine_similarity_with_freshness(
        self, 
        results: List[Dict[str, Any]], 
        similarity_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        유사도와 신선도 점수를 결합하여 최종 점수 계산
        
        Args:
            results (List[Dict[str, Any]]): 검색 결과 목록
            similarity_weight (float, optional): 유사도 가중치. 기본값은 0.7.
            
        Returns:
            List[Dict[str, Any]]: 최종 점수가 추가된 검색 결과 목록
        """
        if not results:
            return []
            
        # 신선도 점수가 없는 경우 계산
        if "freshness_score" not in results[0]:
            results = self.evaluate_data_freshness(results)
            
        combined_results = []
        
        for result in results:
            # 유사도 점수
            similarity = result.get("similarity", 0.0)
            
            # 신선도 점수
            freshness = result.get("freshness_score", 0.0)
            
            # 최종 점수 계산
            final_score = (similarity_weight * similarity) + ((1 - similarity_weight) * freshness)
            
            # 결과에 최종 점수 추가
            result_with_score = result.copy()
            result_with_score["final_score"] = final_score
            
            combined_results.append(result_with_score)
            
        # 최종 점수 기준으로 정렬
        return sorted(combined_results, key=lambda x: x["final_score"], reverse=True)
        
    def find_outdated_entries(
        self, 
        max_age_days: int = 30,
        timestamp_key: str = "timestamp",
        limit: int = 100
    ) -> List[str]:
        """
        오래된 항목 ID 목록 조회
        
        Args:
            max_age_days (int, optional): 최대 나이(일). 기본값은 30.
            timestamp_key (str, optional): 타임스탬프 메타데이터 키. 기본값은 "timestamp".
            limit (int, optional): 최대 결과 수. 기본값은 100.
            
        Returns:
            List[str]: 오래된 항목 ID 목록
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 현재 시간 기준 max_age_days일 이전 타임스탬프 계산
            import datetime as dt
            cutoff_date = datetime.now() - dt.timedelta(days=max_age_days)
            cutoff_timestamp = cutoff_date.isoformat()
            
            # ChromaDB는 날짜 비교를 직접 지원하지 않으므로 모든 항목을 가져와서 필터링
            # 실제 프로덕션 환경에서는 더 효율적인 방법 필요
            all_items = self.collection.get()
            
            if not all_items or not all_items["ids"]:
                return []
                
            outdated_ids = []
            
            for i, id in enumerate(all_items["ids"]):
                metadata = all_items["metadatas"][i] if all_items["metadatas"] else {}
                
                # 타임스탬프 확인
                timestamp = metadata.get(timestamp_key)
                
                if not timestamp:
                    # 타임스탬프가 없는 경우 오래된 것으로 간주
                    outdated_ids.append(id)
                    continue
                    
                try:
                    # 타임스탬프 비교
                    if timestamp < cutoff_timestamp:
                        outdated_ids.append(id)
                except Exception:
                    # 타임스탬프 파싱 오류 시 오래된 것으로 간주
                    outdated_ids.append(id)
                    
                # 최대 결과 수 제한
                if len(outdated_ids) >= limit:
                    break
                    
            return outdated_ids
            
        except Exception as e:
            error_msg = f"오래된 항목 조회 실패: {str(e)}"
            logging.error(error_msg)
            return []
            
    def update_outdated_entry(
        self, 
        id: str, 
        new_content: str = None,
        new_metadata: Dict[str, Any] = None
    ) -> bool:
        """
        오래된 항목 업데이트
        
        Args:
            id (str): 항목 ID
            new_content (str, optional): 새 콘텐츠. 기본값은 None.
            new_metadata (Dict[str, Any], optional): 새 메타데이터. 기본값은 None.
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if not self.collection:
                self._initialize_vector_db()
                
            # 항목 조회
            result = self.collection.get(ids=[id])
            
            if not result or not result["ids"]:
                logging.warning(f"ID '{id}'에 해당하는 항목을 찾을 수 없습니다.")
                return False
                
            # 기존 콘텐츠 및 메타데이터 가져오기
            existing_content = result["documents"][0] if result["documents"] else ""
            existing_metadata = result["metadatas"][0] if result["metadatas"] else {}
            
            # 새 콘텐츠 및 메타데이터 준비
            content_to_update = new_content if new_content is not None else existing_content
            
            # 메타데이터 업데이트
            metadata_to_update = existing_metadata.copy()
            if new_metadata:
                metadata_to_update.update(new_metadata)
                
            # 타임스탬프 업데이트
            metadata_to_update["timestamp"] = datetime.now().isoformat()
            
            # 임베딩 생성
            embedding = self._get_embedding(content_to_update)
            
            # 항목 업데이트
            self.collection.update(
                ids=[id],
                embeddings=[embedding],
                documents=[content_to_update],
                metadatas=[metadata_to_update]
            )
            
            logging.info(f"항목 '{id}' 업데이트 완료")
            return True
            
        except Exception as e:
            error_msg = f"항목 업데이트 실패: {str(e)}"
            logging.error(error_msg)
            return False
            
    def batch_update_outdated_entries(
        self, 
        max_age_days: int = 30,
        batch_size: int = 10,
        update_callback = None
    ) -> Tuple[int, int]:
        """
        오래된 항목 일괄 업데이트
        
        Args:
            max_age_days (int, optional): 최대 나이(일). 기본값은 30.
            batch_size (int, optional): 일괄 처리 크기. 기본값은 10.
            update_callback (callable, optional): 항목 업데이트 콜백 함수. 기본값은 None.
                콜백 함수는 (id, content, metadata) 매개변수를 받아 (new_content, new_metadata) 튜플을 반환해야 함.
            
        Returns:
            Tuple[int, int]: (업데이트된 항목 수, 실패한 항목 수)
        """
        try:
            # 오래된 항목 조회
            outdated_ids = self.find_outdated_entries(max_age_days=max_age_days, limit=batch_size)
            
            if not outdated_ids:
                logging.info("업데이트할 오래된 항목이 없습니다.")
                return (0, 0)
                
            success_count = 0
            failure_count = 0
            
            for id in outdated_ids:
                try:
                    # 항목 조회
                    result = self.collection.get(ids=[id])
                    
                    if not result or not result["ids"]:
                        logging.warning(f"ID '{id}'에 해당하는 항목을 찾을 수 없습니다.")
                        failure_count += 1
                        continue
                        
                    # 기존 콘텐츠 및 메타데이터 가져오기
                    content = result["documents"][0] if result["documents"] else ""
                    metadata = result["metadatas"][0] if result["metadatas"] else {}
                    
                    # 콜백 함수가 있으면 호출하여 새 콘텐츠 및 메타데이터 가져오기
                    new_content = content
                    new_metadata = metadata.copy()
                    
                    if update_callback:
                        try:
                            callback_result = update_callback(id, content, metadata)
                            if callback_result and len(callback_result) == 2:
                                new_content, new_metadata = callback_result
                        except Exception as callback_error:
                            logging.error(f"콜백 함수 실행 실패: {str(callback_error)}")
                            
                    # 항목 업데이트
                    if self.update_outdated_entry(id, new_content, new_metadata):
                        success_count += 1
                    else:
                        failure_count += 1
                        
                except Exception as item_error:
                    logging.error(f"항목 '{id}' 업데이트 실패: {str(item_error)}")
                    failure_count += 1
                    
            logging.info(f"오래된 항목 업데이트 완료: {success_count}개 성공, {failure_count}개 실패")
            return (success_count, failure_count)
            
        except Exception as e:
            error_msg = f"오래된 항목 일괄 업데이트 실패: {str(e)}"
            logging.error(error_msg)
            return (0, 0)