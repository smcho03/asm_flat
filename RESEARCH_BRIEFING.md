# Lensless Holographic Tactile Sensor — Research Briefing

## 연구 개요

He-Ne 레이저(632.8 nm) 기반 **lensless holographic tactile sensor**의 시뮬레이션 및 역문제(inverse problem) 풀기.

핵심 목표: 센서 멤브레인의 변형 h(x,y)를 CMOS 강도 이미지 한 장으로부터 gradient descent로 복원하는 것.

비교 대상: **GelSight** 계열 (LED 비간섭성 조명, 렌즈 카메라, photometric stereo).
- 홀로그래픽 방식은 h < λ/4 = 158 nm 이하 초미세 변형에서 GelSight를 압도함.
- GelSight는 h > ~5 um에서 유리, 158 nm ~ 5 um 사이는 둘 다 어려운 구간.


## 물리 파이프라인

```
1. Phase Modulation  : U0 = exp(i * 4π/λ * h(x,y))
                       (반사 → 왕복 경로 → 2× 위상)

2. ASM Propagation   : U_d = IFFT2[ FFT2(U0) * H(fx, fy) ]
                       H = exp(i*2π*d*√(1/λ²−fx²−fy²))  (band-limited)
                       ref: Matsushima & Shimobaba, Opt. Express 2009

3. CMOS Intensity    : I = |U_d|²  (cropped to sensor size)
```

모든 연산 PyTorch — h에 대해 완전 미분 가능.


## 코드 구조 (`simulation_flat/`)

```
holographic_tactile_sensor.py   ← 메인 시뮬레이터 (HolographicSensor class)
sensor_model.py                 ← 동일 모델 (import용)
sensor_params.py                ← 공유 광학 파라미터
sensor_utils.py                 ← 시각화 유틸
optimize/
  random_pattern_reconstruction.py  ← 현재 주력 스크립트
  gelsight_baseline.py              ← GelSight 포토메트릭 스테레오 비교
  gelsight_mitsuba_compare.py       ← Mitsuba3 물리 기반 GelSight 비교
  gelsight_photometric_compare.py
  lambertian_random_phase.py        ← Lambertian(비간섭성) 광원 비교
  light_source_comparison.py
  aliasing_analysis.py              ← aliasing 원인 분석
  operating_region.py               ← 동작 가능 영역 분석
  reconstruct_sweep.py              ← 거리/진폭별 reconstruction 비교
  hyperparam_search.py              ← 하이퍼파라미터 탐색 (완료)
  optimal_run.py                    ← 최적 파라미터로 단일 실행
  random_iter_snapshots.py          ← 최적화 과정 시각화
  ... (기타 분석 스크립트들)
checks/, sanity/                ← sanity check 스크립트들
output/                         ← 결과 이미지/수치 (gitignore됨)
```

GitHub: `https://github.com/smcho03/asm_flat.git`  
conda 환경: `tactile_pipeline`


## 현재 파라미터 (CPU용 — 너무 작음)

```python
MEM  = 128   # 멤브레인 픽셀 수  → 물리 크기 1.28 mm
CMOS = 256   # CMOS 픽셀 수     → 물리 크기 2.56 mm
GRID = 384   # 시뮬레이션 그리드 (zero-padding)
DIST = 5e-3  # 전파 거리 5 mm
mem_pitch = 10e-6  # 픽셀 피치 10 um
```

**문제**: N이 작아서 aliasing, 해상도 부족 → GPU로 키워야 함.

sensor_params.py에 정의된 풀 해상도 파라미터 (목표치):
```python
mem_res  = 512    # → 물리 크기 5.12 mm
cmos_res = 1024   # → 물리 크기 10.24 mm
grid_res = 1536   # → 물리 크기 15.36 mm
```


## 재구성 알고리즘

```python
# h >= 0 강제 (센서는 안쪽으로만 눌림)
h = raw² * h_scale          # squared reparameterization

# 최적화
optimizer = Adam([raw], lr=3e-3)
scheduler = CosineAnnealingLR(T_max=N_ITER)
loss = MSE(sensor(h), I_target)
N_ITER = 15000
```

- `h_scale = max(2 * A_target, 50e-9)` 로 스케일 조정
- init: `raw = 0.5` (uniform)
- **Adam adaptive lr 덕분에 실질적 lr이 자동 조정됨** → 표면상 lr=3e-3이 맞음
  (이전에 lr=5e-10이었던 건 버그였음, 이미 수정됨)


## 현재까지 실험 결과 (CPU, MEM=128)

### Single Bump (최적 조건)
- PSNR ≈ 22 dB, RMSE ≈ 15 nm (5000 iter, amplitude=200nm, dist=5mm)

### Random Pattern (주요 목표)
| tag | A [nm] | sigma [um] | PSNR [dB] | 비고 |
|-----|--------|------------|-----------|------|
| r1~r3 | 100 | 150 | ~8 dB | CPU, MEM=128 |
| r4 | 200 | 150 | ~8 dB | |
| r5 | 100 | 300 | ~8 dB | |

→ **랜덤 패턴 재구성 PSNR이 너무 낮음. N 키우면 개선될 것으로 예상.**

### GelSight vs Holographic
- h < 158 nm: 홀로그래픽 우세 확인됨
- h > 5 um: GelSight 우세
- Mitsuba3 물리 기반 시뮬레이션으로도 동일 경향 확인


## 남은 이슈 / 할 일

1. **GPU로 MEM=512, CMOS=1024, GRID=1536 으로 재구성 실험**
   - `random_pattern_reconstruction.py`의 MEM/CMOS/GRID 값 수정 후 실행
   - PSNR/RMSE가 어떻게 개선되는지 확인

2. **aliasing 확인**
   - `aliasing_analysis.py`, `operating_region.py` 풀 해상도로 재실행

3. **GelSight 비교 풀 해상도로 재실행**
   - `gelsight_baseline.py`, `gelsight_mitsuba_compare.py`

4. **교수님 피드백 반영 사항**
   - 수평 방향 해상도 증가가 핵심 목표
   - phase wrap 영역 (h > λ/4 = 158 nm)에서 재구성 성능 개선 필요
   - synthesized wavelength 방법 검토 필요 (미구현)


## 알려진 버그 / 주의사항

- Mitsuba3는 경로에 한글 있으면 안 됨 → ASCII 경로로 따로 설치
- Windows cp949 콘솔에서 em dash 출력 시 UnicodeEncodeError → 이미 수정됨
- GelSight 코드에서 flat normal이 (0, -1, 0)임에 주의 (ny < 0)
- `output/` 폴더는 .gitignore에 있어서 git에 없음 — 결과 재현하려면 스크립트 직접 실행
