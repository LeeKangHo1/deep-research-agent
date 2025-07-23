# tests/tools/test_vector_db_manager.py

"""
VectorDBManager 테스트
"""

import os
import uuid
import pytest
import tempfile
from datetime import datetime
from typing import List, Dict, Any

from src.tools.vector_db_manager import VectorDBManager
from src.models.search import SearchResult
from src.models.vector_db import VectorDBEntry


class TestVectorDBManager:
    """VectorDBManager 테스트 클래스"""
    
    @pytest.fixture
    def temp_db_dir(self):
        """임시 DB 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.fixture
    def vector_db_manager(self, temp_db_dir):
        """VectorDBManager 인스턴스 생성"""
        manager = VectorDBManager(
            persist_directory=temp_db_dir,
            collection_name="test_collection"
        )
        return manager
        
    @pytest.fixture
    def sample_search_result(self):
        """샘플 SearchResult 객체 생성"""
        return SearchResult(
            title="테스트 제목",
            url="https://example.com/test",
            content="이것은 테스트 콘텐츠입니다. 벡터 DB 저장 테스트를 위한 샘플 데이터입니다.",
            score=0.85,
            published_date=datetime.now(),
            snippet="테스트 스니펫",
            source_type="web",
            metadata={
                "author": "테스트 작성자",
                "category": "테스트 카테고리"
            }
        )
        
    @pytest.fixture
    def sample_vector_db_entry(self):
        """샘플 VectorDBEntry 객체 생성"""
        return VectorDBEntry(
            content="이것은 VectorDBEntry 테스트 콘텐츠입니다.",
            metadata={
                "source": "테스트 소스",
                "category": "테스트 카테고리"
            },
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],  # 간단한 테스트 벡터
            id=str(uuid.uuid4()),
            timestamp=datetime.now()
        )
        
    def test_initialize_vector_db(self, vector_db_manager):
        """벡터 DB 초기화 테스트"""
        # 초기화 확인
        assert vector_db_manager.client is not None
        assert vector_db_manager.collection is not None
        assert vector_db_manager.collection_name == "test_collection"
        
    def test_preprocess_text_for_embedding(self, vector_db_manager):
        """텍스트 전처리 테스트"""
        # 여러 줄 텍스트
        text = """
        이것은
        여러 줄로 된
        텍스트입니다.
        """
        
        processed = vector_db_manager.preprocess_text_for_embedding(text)
        
        # 여러 줄이 하나로 합쳐졌는지 확인
        assert "\n" not in processed
        assert "이것은 여러 줄로 된 텍스트입니다." == processed
        
        # 긴 텍스트 제한 테스트
        long_text = "a" * 10000
        processed_long = vector_db_manager.preprocess_text_for_embedding(long_text, max_length=100)
        assert len(processed_long) == 100
        
    def test_prepare_metadata(self, vector_db_manager):
        """메타데이터 준비 테스트"""
        # 중첩된 객체가 있는 메타데이터
        metadata = {
            "title": "테스트 제목",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        
        processed = vector_db_manager.prepare_metadata(metadata)
        
        # 타임스탬프 추가 확인
        assert "timestamp" in processed
        
        # 중첩된 객체가 문자열로 변환되었는지 확인
        assert isinstance(processed["nested"], str)
        assert isinstance(processed["list"], str)
        
    def test_store_search_result(self, vector_db_manager, sample_search_result):
        """검색 결과 저장 테스트"""
        # 검색 결과 저장
        id = vector_db_manager.store_search_result(sample_search_result, query="테스트 쿼리")
        
        # ID가 생성되었는지 확인
        assert id is not None
        assert isinstance(id, str)
        
        # 저장된 항목 조회
        result = vector_db_manager.collection.get(ids=[id])
        
        # 저장된 항목 확인
        assert result["ids"][0] == id
        assert sample_search_result.content in result["documents"][0]
        assert result["metadatas"][0]["title"] == sample_search_result.title
        assert result["metadatas"][0]["query"] == "테스트 쿼리"
        
    def test_store_multiple_search_results(self, vector_db_manager, sample_search_result):
        """여러 검색 결과 저장 테스트"""
        # 여러 검색 결과 생성
        results = [
            sample_search_result,
            SearchResult(
                title="두 번째 테스트",
                url="https://example.com/test2",
                content="두 번째 테스트 콘텐츠입니다.",
                score=0.75
            )
        ]
        
        # 검색 결과 일괄 저장
        ids = vector_db_manager.store_multiple_search_results(results, query="일괄 테스트")
        
        # ID가 생성되었는지 확인
        assert len(ids) == 2
        
        # 저장된 항목 조회
        for id in ids:
            result = vector_db_manager.collection.get(ids=[id])
            assert result["ids"][0] == id
            assert result["metadatas"][0]["query"] == "일괄 테스트"
            
    def test_store_vector_db_entry(self, vector_db_manager, sample_vector_db_entry):
        """VectorDBEntry 저장 테스트"""
        # VectorDBEntry 저장
        id = vector_db_manager.store_vector_db_entry(sample_vector_db_entry)
        
        # ID가 생성되었는지 확인
        assert id == sample_vector_db_entry.id
        
        # 저장된 항목 조회
        result = vector_db_manager.collection.get(ids=[id])
        
        # 저장된 항목 확인
        assert result["ids"][0] == id
        assert sample_vector_db_entry.content in result["documents"][0]
        assert result["metadatas"][0]["source"] == sample_vector_db_entry.metadata["source"]
        
    def test_update_metadata(self, vector_db_manager, sample_search_result):
        """메타데이터 업데이트 테스트"""
        # 검색 결과 저장
        id = vector_db_manager.store_search_result(sample_search_result)
        
        # 메타데이터 업데이트
        new_metadata = {"updated": True, "category": "업데이트된 카테고리"}
        success = vector_db_manager.update_metadata(id, new_metadata)
        
        # 업데이트 성공 확인
        assert success
        
        # 업데이트된 항목 조회
        result = vector_db_manager.collection.get(ids=[id])
        
        # 업데이트된 메타데이터 확인
        assert result["metadatas"][0]["updated"] == True
        assert result["metadatas"][0]["category"] == "업데이트된 카테고리"
        # 기존 메타데이터는 유지되어야 함
        assert result["metadatas"][0]["title"] == sample_search_result.title
        
    def test_search_by_text(self, vector_db_manager, sample_search_result):
        """텍스트 기반 검색 테스트"""
        # 검색 결과 저장
        vector_db_manager.store_search_result(sample_search_result)
        
        # 텍스트 기반 검색
        results = vector_db_manager.search_by_text("테스트 콘텐츠", top_k=5)
        
        # 검색 결과 확인
        assert len(results) > 0
        assert "테스트 콘텐츠" in results[0]["content"]
        assert "similarity" in results[0]
        assert results[0]["similarity"] > 0.0
        
    def test_search_with_threshold(self, vector_db_manager):
        """임계값 기반 검색 테스트"""
        # 여러 검색 결과 저장
        vector_db_manager.store_content_with_metadata(
            "첫 번째 테스트 콘텐츠입니다.",
            {"category": "테스트"}
        )
        vector_db_manager.store_content_with_metadata(
            "두 번째 테스트 콘텐츠입니다.",
            {"category": "테스트"}
        )
        vector_db_manager.store_content_with_metadata(
            "완전히 다른 내용의 콘텐츠입니다.",
            {"category": "다른 카테고리"}
        )
        
        # 높은 임계값으로 검색
        results_high = vector_db_manager.search_by_text("테스트 콘텐츠", threshold=0.9)
        
        # 낮은 임계값으로 검색
        results_low = vector_db_manager.search_by_text("테스트 콘텐츠", threshold=0.1)
        
        # 임계값에 따른 결과 수 비교
        assert len(results_high) <= len(results_low)
        
    def test_search_with_metadata_filter(self, vector_db_manager):
        """메타데이터 필터 검색 테스트"""
        # 여러 검색 결과 저장
        vector_db_manager.store_content_with_metadata(
            "첫 번째 테스트 콘텐츠입니다.",
            {"category": "카테고리A"}
        )
        vector_db_manager.store_content_with_metadata(
            "두 번째 테스트 콘텐츠입니다.",
            {"category": "카테고리B"}
        )
        
        # 메타데이터 필터로 검색
        results = vector_db_manager.search_by_text(
            "테스트 콘텐츠",
            metadata_filter={"category": "카테고리A"}
        )
        
        # 필터링된 결과 확인
        assert len(results) > 0
        assert results[0]["metadata"]["category"] == "카테고리A"
        
    def test_search_similar_to_id(self, vector_db_manager):
        """ID 기반 유사 항목 검색 테스트"""
        # 여러 검색 결과 저장
        id1 = vector_db_manager.store_content_with_metadata(
            "인공지능에 관한 첫 번째 콘텐츠입니다.",
            {"category": "AI"}
        )
        vector_db_manager.store_content_with_metadata(
            "인공지능에 관한 두 번째 콘텐츠입니다.",
            {"category": "AI"}
        )
        vector_db_manager.store_content_with_metadata(
            "완전히 다른 주제의 콘텐츠입니다.",
            {"category": "기타"}
        )
        
        # ID 기반 유사 항목 검색
        results = vector_db_manager.search_similar_to_id(id1, top_k=2)
        
        # 검색 결과 확인
        assert len(results) > 0
        assert "인공지능" in results[0]["content"]
        
    def test_filter_by_similarity(self, vector_db_manager):
        """유사도 기반 필터링 테스트"""
        # 검색 결과 생성
        results = [
            {"id": "1", "content": "콘텐츠1", "similarity": 0.9},
            {"id": "2", "content": "콘텐츠2", "similarity": 0.7},
            {"id": "3", "content": "콘텐츠3", "similarity": 0.5},
            {"id": "4", "content": "콘텐츠4", "similarity": 0.3}
        ]
        
        # 유사도 기반 필터링
        filtered = vector_db_manager.filter_by_similarity(results, threshold=0.6)
        
        # 필터링 결과 확인
        assert len(filtered) == 2
        assert filtered[0]["id"] == "1"
        assert filtered[1]["id"] == "2"
        
    def test_rank_by_similarity(self, vector_db_manager):
        """유사도 기반 정렬 테스트"""
        # 검색 결과 생성 (순서 섞음)
        results = [
            {"id": "2", "content": "콘텐츠2", "similarity": 0.7},
            {"id": "4", "content": "콘텐츠4", "similarity": 0.3},
            {"id": "1", "content": "콘텐츠1", "similarity": 0.9},
            {"id": "3", "content": "콘텐츠3", "similarity": 0.5}
        ]
        
        # 유사도 기반 정렬 (내림차순)
        ranked = vector_db_manager.rank_by_similarity(results)
        
        # 정렬 결과 확인
        assert len(ranked) == 4
        assert ranked[0]["id"] == "1"  # 0.9
        assert ranked[1]["id"] == "2"  # 0.7
        assert ranked[2]["id"] == "3"  # 0.5
        assert ranked[3]["id"] == "4"  # 0.3
        
        # 유사도 기반 정렬 (오름차순)
        ranked_asc = vector_db_manager.rank_by_similarity(results, reverse=False)
        
        # 정렬 결과 확인
        assert ranked_asc[0]["id"] == "4"  # 0.3
        assert ranked_asc[3]["id"] == "1"  # 0.9
        
    def test_calculate_similarity(self, vector_db_manager):
        """유사도 계산 테스트"""
        # 두 벡터 간의 유사도 계산
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        vector3 = [1.0, 0.0, 0.0]  # vector1과 동일
        
        # 직교 벡터 (유사도 0)
        similarity1 = vector_db_manager.calculate_similarity(vector1, vector2)
        assert similarity1 == 0.0
        
        # 동일 벡터 (유사도 1)
        similarity2 = vector_db_manager.calculate_similarity(vector1, vector3)
        assert similarity2 == 1.0
        
        # 유사한 벡터
        vector4 = [0.8, 0.2, 0.0]
        similarity3 = vector_db_manager.calculate_similarity(vector1, vector4)
        assert 0.0 < similarity3 < 1.0
        
    def test_evaluate_data_freshness(self, vector_db_manager):
        """데이터 신선도 평가 테스트"""
        # 검색 결과 생성
        results = [
            {
                "id": "1",
                "content": "콘텐츠1",
                "metadata": {
                    "timestamp": (datetime.now().isoformat())
                }
            },
            {
                "id": "2",
                "content": "콘텐츠2",
                "metadata": {
                    "timestamp": (datetime.now().replace(year=datetime.now().year - 1).isoformat())
                }
            }
        ]
        
        # 신선도 평가
        results_with_freshness = vector_db_manager.evaluate_data_freshness(results)
        
        # 신선도 점수 확인
        assert len(results_with_freshness) == 2
        assert "freshness_score" in results_with_freshness[0]
        assert "freshness_score" in results_with_freshness[1]
        
        # 최신 데이터의 신선도 점수가 더 높아야 함
        assert results_with_freshness[0]["freshness_score"] > results_with_freshness[1]["freshness_score"]
        
    def test_filter_by_freshness(self, vector_db_manager):
        """신선도 기반 필터링 테스트"""
        # 신선도 점수가 있는 검색 결과 생성
        results = [
            {"id": "1", "content": "콘텐츠1", "freshness_score": 0.9},
            {"id": "2", "content": "콘텐츠2", "freshness_score": 0.7},
            {"id": "3", "content": "콘텐츠3", "freshness_score": 0.5},
            {"id": "4", "content": "콘텐츠4", "freshness_score": 0.3}
        ]
        
        # 신선도 기반 필터링
        filtered = vector_db_manager.filter_by_freshness(results, min_freshness=0.6)
        
        # 필터링 결과 확인
        assert len(filtered) == 2
        assert filtered[0]["id"] == "1"
        assert filtered[1]["id"] == "2"
        
    def test_combine_similarity_with_freshness(self, vector_db_manager):
        """유사도와 신선도 결합 테스트"""
        # 유사도와 신선도 점수가 있는 검색 결과 생성
        results = [
            {"id": "1", "content": "콘텐츠1", "similarity": 0.9, "freshness_score": 0.5},
            {"id": "2", "content": "콘텐츠2", "similarity": 0.7, "freshness_score": 0.9},
            {"id": "3", "content": "콘텐츠3", "similarity": 0.5, "freshness_score": 0.7}
        ]
        
        # 유사도 가중치 0.7로 결합
        combined = vector_db_manager.combine_similarity_with_freshness(results, similarity_weight=0.7)
        
        # 결합 결과 확인
        assert len(combined) == 3
        assert "final_score" in combined[0]
        
        # 최종 점수 계산 확인
        # 콘텐츠1: 0.7 * 0.9 + 0.3 * 0.5 = 0.63 + 0.15 = 0.78
        # 콘텐츠2: 0.7 * 0.7 + 0.3 * 0.9 = 0.49 + 0.27 = 0.76
        # 콘텐츠3: 0.7 * 0.5 + 0.3 * 0.7 = 0.35 + 0.21 = 0.56
        assert combined[0]["id"] == "1"  # 최고 점수
        assert combined[1]["id"] == "2"
        assert combined[2]["id"] == "3"
        
    def test_update_outdated_entry(self, vector_db_manager):
        """오래된 항목 업데이트 테스트"""
        # 항목 저장
        id = vector_db_manager.store_content_with_metadata(
            "원본 콘텐츠입니다.",
            {"category": "테스트", "timestamp": "2020-01-01T00:00:00"}
        )
        
        # 항목 업데이트
        success = vector_db_manager.update_outdated_entry(
            id,
            new_content="업데이트된 콘텐츠입니다.",
            new_metadata={"updated": True}
        )
        
        # 업데이트 성공 확인
        assert success
        
        # 업데이트된 항목 조회
        result = vector_db_manager.collection.get(ids=[id])
        
        # 업데이트된 내용 확인
        assert "업데이트된 콘텐츠입니다." in result["documents"][0]
        assert result["metadatas"][0]["updated"] == True
        assert result["metadatas"][0]["category"] == "테스트"  # 기존 메타데이터 유지
        assert "timestamp" in result["metadatas"][0]  # 타임스탬프 업데이트
        
    def test_find_outdated_entries(self, vector_db_manager, monkeypatch):
        """오래된 항목 찾기 테스트"""
        # datetime.timedelta 모듈 패치
        import datetime as dt
        monkeypatch.setattr(dt, "timedelta", lambda days: dt.timedelta(days=days))
        
        # 최신 항목 저장
        id1 = vector_db_manager.store_content_with_metadata(
            "최신 콘텐츠입니다.",
            {"timestamp": datetime.now().isoformat()}
        )
        
        # 오래된 항목 저장 (1년 전)
        old_date = datetime.now().replace(year=datetime.now().year - 1)
        id2 = vector_db_manager.store_content_with_metadata(
            "오래된 콘텐츠입니다.",
            {"timestamp": old_date.isoformat()}
        )
        
        # 타임스탬프가 없는 항목 저장
        id3 = vector_db_manager.store_content_with_metadata(
            "타임스탬프가 없는 콘텐츠입니다.",
            {"category": "테스트"}
        )
        
        # 오래된 항목 찾기 (30일 기준)
        outdated_ids = vector_db_manager.find_outdated_entries(max_age_days=30)
        
        # 오래된 항목 확인
        assert len(outdated_ids) >= 2
        assert id2 in outdated_ids  # 1년 전 항목
        assert id3 in outdated_ids  # 타임스탬프 없는 항목
        assert id1 not in outdated_ids  # 최신 항목
        
    def test_batch_update_outdated_entries(self, vector_db_manager, monkeypatch):
        """오래된 항목 일괄 업데이트 테스트"""
        # datetime.timedelta 모듈 패치
        import datetime as dt
        monkeypatch.setattr(dt, "timedelta", lambda days: dt.timedelta(days=days))
        
        # 오래된 항목 여러 개 저장
        old_date = datetime.now().replace(year=datetime.now().year - 1)
        for i in range(3):
            vector_db_manager.store_content_with_metadata(
                f"오래된 콘텐츠 {i}입니다.",
                {"timestamp": old_date.isoformat(), "index": i}
            )
            
        # 업데이트 콜백 함수
        def update_callback(id, content, metadata):
            index = metadata.get("index", 0)
            return f"업데이트된 콘텐츠 {index}입니다.", {"updated": True, "index": index}
            
        # 오래된 항목 일괄 업데이트
        success_count, failure_count = vector_db_manager.batch_update_outdated_entries(
            max_age_days=30,
            batch_size=5,
            update_callback=update_callback
        )
        
        # 업데이트 결과 확인
        assert success_count == 3
        assert failure_count == 0
        
        # 업데이트된 항목 확인
        results = vector_db_manager.search_by_text("업데이트된 콘텐츠", top_k=5)
        assert len(results) == 3
        for result in results:
            assert result["metadata"]["updated"] == True
            assert "timestamp" in result["metadata"]  # 타임스탬프 업데이트