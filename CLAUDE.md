# Pixlato — 프로젝트 지침

## 프로젝트 개요

Pixlato는 U2-Net 기반 배경제거 + K-Means 픽셀레이션을 결합한 Windows 전용 픽셀아트 변환 앱이다.
- **언어**: Python 3.x
- **UI**: CustomTkinter
- **GPU 가속**: PyTorch (CUDA) / onnxruntime-directml (DirectML)
- **배포**: PyInstaller 단일 실행파일

---

## 아키텍처 핵심 규칙

### 레이어 구조 (위반 금지)

```
ui/app.py                   ← UI 레이어. 비즈니스 로직 포함 금지
    ↓ 호출
core/processor.py           ← 공개 API + EngineDispatcher (백엔드 선택)
    ↓ 위임
core/processor_torch.py     ← GPU (PyTorch/CUDA) 구현
core/processor_numpy.py     ← CPU (NumPy) 구현
```

- `app.py`는 `processor.py`의 공개 함수만 호출한다. `processor_torch.py`를 직접 호출하지 않는다.
- `processor_torch.py`와 `processor_numpy.py`는 서로 의존하지 않는다.

### EngineDispatcher 규칙

- 모든 백엔드 선택(GPU/CPU)은 `EngineDispatcher`를 통한다.
- 개별 함수가 하드웨어를 직접 탐지하는 코드를 추가하지 않는다.
- `_build_rembg_providers()`: ONNX provider 우선순위의 단일 진실 공급원. 다른 곳에서 provider 목록을 직접 만들지 않는다.

### GPU 폴백 원칙

모든 GPU 경로는 반드시 CPU 폴백을 보장한다.

```python
# 올바른 패턴
try:
    # GPU 경로
    ...
except Exception as e:          # ImportError만 잡지 말 것
    debug_log(f"... {e}")
    # CPU 폴백
    ...
```

---

## Phase 작업 시 주의사항

### 공통

- Phase 작업은 반드시 worktree에서 진행한다. master에 직접 커밋하지 않는다.
- 전역 CLAUDE.md의 AI-Driven SDLC 10단계를 모든 Phase에 적용한다.
- Phase 완료 후 `agent_docs/implementation_plan.md`에 날짜와 함께 완료 처리한다.

### 커밋 메시지 형식

```
<type>(Phase-<N>): <요약>

- 변경 항목 1
- 변경 항목 2

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
```

type: `feat` / `fix` / `refactor` / `chore` / `docs`

### 테스트 환경 한계

- 개발 환경에 `onnxruntime`이 미설치된 경우가 있다. `_build_rembg_providers()`는 이 경우 CPU 폴백을 반환하며, 이는 정상 동작이다.
- GPU 경로의 실제 동작은 `onnxruntime-directml` 설치된 앱 실행 환경에서 별도 검증이 필요하다.

---

## 알려진 기술 부채 (Known Technical Debt)

| 항목 | 위치 | 우선순위 | 설명 |
|---|---|---|---|
| Dead Code | `processor_torch.py#remove_background_ai_torch` | LOW | Deprecated 처리 완료 (Phase-59). Phase-60에서 제거 예정 |
| 세션 이중화 | `processor.py:REMBG_SESSION` / `processor_torch.py:_REMBG_SESSION` | LOW | 동일 모델 2개 세션. `_REMBG_SESSION`은 Deprecated 함수 전용 — Phase-60 함수 제거 시 함께 삭제 |
| `_build_rembg_providers` 캐싱 없음 | `processor.py:84` | LOW | 세션 리셋 시 재호출됨. 성능 영향 미미하나 방어적 캐싱 검토 |
| Interactive 모드 UI 미구현 | `ui/app.py` | MEDIUM | `bg_seeds` 수집 로직 없어 실제 사용 불가 |
| `_update_secondary_ui` 락 없음 | `ui/app.py#_update_secondary_ui` | LOW | `self.original_size`를 락 없이 읽음. 스레드 재진입 시 stale `(0,0)` 반환 가능. Phase-60 이전에 `hasattr` 가드 추가 검토 |
| Plugin sandbox `getattr`/`setattr` | `core/plugin_engine.py#_load_plugin` | LOW | exec 샌드박스 내 `getattr`으로 `__class__.__mro__` 체인 순회 가능. 별도 보안 강화 Phase에서 검토 |
| Plugin torch 탐지 반복 실행 | `core/plugin_engine.py#_load_plugin` | LOW | 플러그인 로드마다 `import torch` 시도. `sys.modules` 캐시로 실 I/O는 없으나, 클래스 레벨 탐지 플래그로 단일화 권장 |

---

## Git 규칙

- **Remote**: `https://github.com/Feel-o-sofa/Pixlato.git`
- **기본 브랜치**: `master`
- **worktree 경로**: `.claude/worktrees/<branch-name>` (`.gitignore`에 등록됨)
- force push 금지. 항상 rebase → ff-only merge 순서로 진행한다.
