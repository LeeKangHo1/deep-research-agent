# Deep Research Chatbot

LangChain과 LangGraph를 활용한 멀티 에이전트 시스템 기반 심층 연구 챗봇입니다.

## 프로젝트 구조

```
deep-research-chatbot/
├── agents/              # AI 에이전트들
├── api/                 # FastAPI 엔드포인트
├── config/              # 설정 파일들
├── models/              # 데이터 모델
├── tools/               # 외부 도구 및 유틸리티
├── workflows/           # LangGraph 워크플로우
├── data/                # 데이터 저장소 (자동 생성)
├── logs/                # 로그 파일 (자동 생성)
├── main.py              # 메인 애플리케이션
├── requirements.txt     # 의존성 목록
└── .env.example         # 환경 변수 템플릿
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키들을 설정하세요
```

### 3. 애플리케이션 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --reload
```

## 주요 기능

- **멀티 에이전트 시스템**: 연구, 분석, 종합, 검증 에이전트들의 협력
- **심층 연구**: 다각도 정보 수집 및 분석
- **대화형 인터페이스**: 자연스러운 대화를 통한 연구 요청
- **벡터 데이터베이스**: 이전 연구 결과 활용
- **확장 가능한 아키텍처**: 새로운 에이전트 및 도구 추가 용이

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 개발 상태

현재 프로젝트 구조 및 기본 설정이 완료되었습니다. 
다음 단계에서 각 컴포넌트들을 순차적으로 구현할 예정입니다.