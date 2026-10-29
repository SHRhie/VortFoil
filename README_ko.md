[English](README.md) | **한국어**

이 문서는 README.md의 한국어 번역입니다. 두 문서가 다르면 영어판(README.md)이 기준입니다.

# VortFoil

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23336871-blue)](https://doi.org/10.5281/zenodo.23336871)

의도된 용도와 한계는 [`MODEL_CARD.md`](MODEL_CARD.md)를, 릴리스 노트는 [`CHANGELOG.md`](CHANGELOG.md)를 참고하십시오.

**"VortFoil: A Vortex-Panel-Informed Neural Operator for Steady Airfoil Flow Prediction."** 논문에
딸린 코드, 학습된 가중치, 봉인된 결과 파일, 그림입니다.

VortFoil은 고전적인 와류 패널 공기역학을 신경망 오퍼레이터의 순전파 안에 내장합니다. 네트워크가
변화율 함수를 예측하고, 그 닫힌 형태의 역도함수가 표면 와류 시트가 됩니다. 선형 사영이 이산 쿠타
조건을 정확히 부과합니다. 표면 압력, 순환 양력 `Cl^γ`, ¼ 시위 모멘트, 정체점, 뒷전의 경계층 가장자리
속도는 이 시트의 패널법 판독값이며, 따로 학습되는 양이 아닙니다. 두 번째 헤드 묶음이 적분 경계층을
예측합니다. 배포된 두 힘 계수는 순수한 시트 판독값이 *아닙니다*. 형상 항력은 예측된 뒷전 상태에 대한
릿지 회귀 닫힘식이고, 배포된 양력은 순환 판독값에 같은 상태에서 읽어 낸 릿지 회귀 점성 캠버 감소
보정을 더한 값입니다. 형상 비의존 유동장 디코더는 외부 솔버가 아니라 시트의 순환과 경계층의 뒷전 상태를
조건으로 하여 2차원 RANS 유동장을 복원합니다. **이 저장소는 확정된 표면 모델 하나, 이를 조건으로 하는
유동장 디코더, 그리고 표면 계수에 대한 고정된 RANS 보정을 제공하며**, 원고에서 비교하는 디코더들도 함께 제공합니다. 형상만으로 계수를
회귀하는 모델은 없으며, 세 단계 어디에서도 추론 시점에 XFOIL이나 CFD 프로세스를 실행하지 않습니다.
이는 외부 솔버 호출에 관한 진술일 뿐, 적합이나 선형대수에 관한 진술이 아닙니다. 두 힘 닫힘식은 배포
시점에 고정된 경험적 적합이며, 유동장 경로의 물리 페인트(physics paint)는 여전히 모델 내부에서
결정론적인 패널 풀이를 수행합니다.

원고의 초록은 [`paper/abstract.md`](paper/abstract.md)를 참고하십시오. 정본은 원고이며, 이곳의 초록은 미러(사본)이므로
이곳에서 직접 수정하지 않습니다.

> **상태.** 버전 2.1.0은 수정된 원고 *VortFoil: A Vortex-Panel-Informed Neural Operator for Steady Airfoil Flow Prediction*
> (S.-H. Rhie, G. Kim, S. Lim, J. Lee)에 대응합니다. 유동장 디코더는 AirfRANS와 비-NACA UniFoil 익형 16,923개를 함께 사용해
> 학습했으며, 표면 브랜치의 RANS 계수 보정이 추가되었습니다(`CHANGELOG.md`).
> 버전 2.0.0은 2026-10-13에 GitHub과 PyPI로 릴리스되었습니다. 모든 버전은 Zenodo의
> [10.5281/zenodo.23336871](https://doi.org/10.5281/zenodo.23336871)에 보관됩니다.
> 라이선스는 GPL-3.0-or-later입니다(아래의 *라이선스* 참조).

## 디렉터리 구조

```
code/             두 개의 스모크 테스트와 경로 도우미(vortfoil_paths.py)
code/model/       OP1(시트)과 OP2(경계층) 오퍼레이터 정의, ADA-F 구성 요소,
                  그리고 항력 닫힘식 학습 코드
code/field/       유동장 디코더의 상태 제공 모듈, 학습/평가 구동 스크립트, 유동장 채점 스크립트
code/reproduce/   보고된 수치의 근거가 되는 측정 스크립트
code/figures/     원고용 그림 생성 스크립트
weights/          배포되는 23개 가중치 파일(아래 표 참조)
data/eval/        XFOIL 평가 타깃(학습 데이터 아님)
results/          원고 수치의 원본인 봉인된 기록
evaluation_runs/  XFOIL 조건부 디코더 실행의 봉인된 사례별 지표(유동장 채점 스크립트의 기준)
figures/          원고에 실린 그대로의 그림과 저장된 수치
paper/            원고 초록
```

파일별 용도 전체, 경로를 재정의하는 `VORTFOIL_*` 환경 변수, 그리고 여기에 제공되지 *않는* 연구 트리 산출물의
명명 규칙은 모두 [`code/README_code.md`](code/README_code.md)에 있습니다. `code/`에서 무엇이든 실행하기 전에
먼저 읽으십시오.

## 모델

1. **시트 오퍼레이터(OP1)** — `weights/op1_supervised_eps0.50.scr_bd128.npz`. 조건 벡터
   `[CST16, sin α, cos α, log Re] ∈ R^19` → 고정된 241개 노드 루프 위의 ADA-F 와류 시트 강도 γ
   (브랜치 128×3 / 트렁크 128×2, N_m = 24, 쿠타 사영). 모든 표면량은 γ의 판독값입니다. Cp, 순환 양력
   Cl^γ = 2∮γ ds, Cm(압력 적분), 정체점(접선 속도의 유일한 부호 변화), 그리고 뒷전의 u_e입니다.
2. **경계층 오퍼레이터(OP2)** — `weights/op2_m325_BT.npz`. 네 개의 ADA-F 헤드(두 표면의 θ, δ\*; 표면별
   균일한 x/c 격자 121개 노드, N_m = 16)이며, 게이트를 통과한 XFOIL 경계층 행 325,370개로 학습했습니다.
   시트를 입력으로 받지 않습니다.
3. **항력 닫힘식** — `weights/cd_closure_poly2.npz`. `code/model/staged/common/wake_ode_tf.py`에 구현되어
   있습니다(기본값 `closure="fitted_poly2"`). log10 Cd = wᵀφ(log θ_TE, H_TE, u_e,TE)로, 학습 데크에서
   릿지 회귀로 적합한 열 개의 계수를 사용하고 결과는 `[1e-4, 0.1]` 범위로 클리핑합니다. 후류를 전진 적분하는
   Squire–Young 방식을 대체합니다(아래 결과 요약 참조).
4. **양력 닫힘식** — `weights/lift_closure_poly2.npz`. `code/model/staged/common/lift_closure_tf.py`
   (`cl_closed_tf`)에 구현되어 있습니다. Cl = Cl^γ + wᵀφ(Cl^γ, θ_u,TE, θ_l,TE, δ\*_u,TE, δ\*_l,TE, u_e,TE)이며,
   예측된 뒷전 상태에 대한 2차 다항식입니다(같은 학습 행 325,370개에서 릿지 회귀로 적합해 고정한 28개 계수이며,
   윗면과 아랫면이 따로 들어갑니다). 시트의 순환 양력에 대한 점성 캠버 감소 보정이며, 배포된 Cl은
   닫힘식을 적용한 이 값입니다. 아래의 유동장 디코더는 여전히 보정되지 않은 Cl^γ를 입력으로 받습니다.
5. **유동장 디코더** — `weights/field_decoder_cmx_s0.npz`, 배포된 디코더입니다(2차원 보기와 3차원 날개의 RANS 상태). 이는 형상 비의존 RANS 유동장
   디코더(폭 248의 다섯 층 비잔차 트렁크, 네 대역 푸리에 특징, 실제 표면 점군에 대한 PointNet 형상 잠재 벡터
   z ∈ R^16, 물리 페인트 입력; ≈305k 매개변수)로서 AirfRANS k–ω SST 사례 780개(800개 사례 학습 분할에서
   검증용으로 20개를 제외한 것)와 비-NACA UniFoil 익형 16,923개(완전 난류 Spalart–Allmaras, M ≤ 0.2;
   익형 150개는 홀드아웃)를 함께 사용해 학습했으며, 난류 모델 입력은 없습니다. 세 번의 학습 실행 가운데 배포된
   실행은 검증 사례 20개에서 표준화된 네 채널 MSE의 평균이 0.00257로 가장 낮습니다(실행 1과 2는 0.00458과
   0.00330; `results/cotrain_score_cmx_20261015T023434Z.json`, `selection`). 원고의 제거 실험용 디코더도 함께
   제공합니다. 아래에 설명하는 AirfRANS 전용 디코더들(릴리스 2.0.0은 `field_decoder_f0cm_s1.npz`를 배포함),
   배포된 디코더와 같은 수의 최적화 스텝으로 학습한 AirfRANS 전용
   디코더(`field_decoder_sml2_s*.npz`, 실행 2 선택; `results/stepmatch_score_20261015T070933Z.json`), 그리고
   UniFoil 익형 1,346개를 함께 사용해 학습한 디코더(`field_decoder_cmd_s*.npz`, 실행 2 선택;
   `results/cotrain_score_cmd_20261015T032838Z.json`)입니다. AirfRANS 전용 오퍼레이터 조건부 디코더
   `weights/field_decoder_f0cm_s1.npz`는 구조가 같으며 AirfRANS 사례 780개만으로 학습했습니다. 두 조건 입력은 XFOIL
   호출이 아니라 `code/field/state_provider.py`를 통해 OP1과 OP2에서 얻습니다. 하나는 페인트 순환 스케일
   Cl^γ(보정 전)이고, 다른 하나는 박리 스칼라 `s_sep = clip((max(H_TE^u, H_TE^l) − 2)/8, 0, 1)`
   (`s_rule="teH"`)로, 표면별 뒷전 형상 계수 두 값 중 큰 쪽이며 이 디코더의 학습 행과 모든 평가에 쓰입니다.
   비교용 디코더 `weights/field_decoder_f0a_s2.npz`는 구조와 학습 레시피가 같고 XFOIL의 입력으로
   학습되고 평가됩니다(XFOIL 조건부 디코더). 이 두 디코더 각각의 세 번의 학습 실행 가운데 선택된 실행은 학습기의
   20개 사례 검증 홀드아웃에서 표준화된 네 채널 MSE의 평균이 가장 낮습니다. 오퍼레이터 조건부 실행 1은
   0.00262(`results/f0cm_holdout_rescore.json`), XFOIL 조건부 실행 2는
   0.00372(`results/f0a_holdout_rescore_20260930.jsonl`)입니다. 다른 실행도 함께
   제공합니다(`results/field_deployed_weights.json`, `selection.*`). `weights/airfrans_surface_tokens.npz`(형상 잠재
   벡터용 표면 점 토큰)와 `weights/airfrans_train_stats.npz`(출력 정규화)는 디코더가 필요로 하는 입력이며,
   별도의 모델이 아닙니다. 여기의 모든 디코더는 같은 두 조건 입력을 받습니다.
6. **RANS 계수 보정** — `weights/rans_correction_nn.npz`(패키지에도 사본이 있음). 모든 오퍼레이터와 닫힘식을
   고정한 상태에서, 5-32-32-3 tanh 네트워크 다섯 개가 표면 브랜치의 (C_l, ln C_d, C_m, α, log₁₀ Re)를
   RANS 수준의 계수 C_l + f_l, C_d·exp(f_d), C_m + f_m으로 변환하며, 그 평균을 사용합니다. 형상이나 경계층 상태는
   입력받지 않습니다. 적합 전에 고정한 프로토콜(`results/rans_correction_nn_20261015_PRESPEC.md`)에 따라
   AirfRANS 학습 사례 780개(k–ω SST)로 적합했습니다. `vf.predict(..., rans=True)`는 `cl_rans`, `cd_rans`,
   `cm_rans`를 반환하고, `vortfoil predict --rans`는 이를 출력합니다. 적합 범위(α −5°에서 15°, Re 2×10⁶에서
   6×10⁶)를 벗어나면 `vortfoil.RangeWarning`을 발생시킵니다.

## 가중치

| 파일 | 역할 | sha256 (앞 12자) |
|---|---|---|
| `op1_supervised_eps0.50.scr_bd128.npz` | OP1 시트 오퍼레이터 | `3bc4baaba1cc` |
| `op2_m325_BT.npz` | OP2 경계층 오퍼레이터 | `a21ce5116b0b` |
| `cd_closure_poly2.npz` | 항력 닫힘식 계수(`fitted_poly2`) | `1e37b6e3dfb4` |
| `lift_closure_poly2.npz` | 양력 닫힘식 계수(`cl_closed_tf`) | `788f7c73efb2` |
| `field_decoder_f0a_s0.npz` | 유동장 디코더, XFOIL 조건부, 실행 0 | `d90b88a5757e` |
| `field_decoder_f0a_s1.npz` | 유동장 디코더, XFOIL 조건부, 실행 1 | `118face2a9e8` |
| `field_decoder_f0a_s2.npz` | 유동장 디코더, XFOIL 조건부, 실행 2(선택됨) | `1440f8f4dbd3` |
| `field_decoder_f0cm_s0.npz` | 유동장 디코더, AirfRANS 전용 오퍼레이터 조건부, 실행 0 | `e1da6228e827` |
| `field_decoder_f0cm_s1.npz` | 유동장 디코더, AirfRANS 전용 오퍼레이터 조건부, 실행 1(선택됨; 2.0.0에서 배포) | `5e72da181372` |
| `field_decoder_f0cm_s2.npz` | 유동장 디코더, AirfRANS 전용 오퍼레이터 조건부, 실행 2 | `fed72563ca1e` |
| `field_decoder_cmx_s0.npz` | 유동장 디코더, AirfRANS + UniFoil 16,923개, 실행 0(배포됨) | `3a8a14810969` |
| `field_decoder_cmx_s1.npz` | 유동장 디코더, AirfRANS + UniFoil 16,923개, 실행 1 | `fdeaa79f5701` |
| `field_decoder_cmx_s2.npz` | 유동장 디코더, AirfRANS + UniFoil 16,923개, 실행 2 | `216f4c5f77d1` |
| `field_decoder_sml2_s0.npz` | 유동장 디코더, AirfRANS 전용, 최적화 스텝 수 일치(341 에포크), 실행 0 | `6c5f4f98f66f` |
| `field_decoder_sml2_s1.npz` | 유동장 디코더, AirfRANS 전용, 최적화 스텝 수 일치, 실행 1 | `fe6cb58df767` |
| `field_decoder_sml2_s2.npz` | 유동장 디코더, AirfRANS 전용, 최적화 스텝 수 일치, 실행 2(선택됨) | `677339fac1be` |
| `field_decoder_cmd_s0.npz` | 유동장 디코더, AirfRANS + UniFoil 1,346개, 실행 0 | `ea4afc5302cc` |
| `field_decoder_cmd_s1.npz` | 유동장 디코더, AirfRANS + UniFoil 1,346개, 실행 1 | `27061f0007bd` |
| `field_decoder_cmd_s2.npz` | 유동장 디코더, AirfRANS + UniFoil 1,346개, 실행 2(선택됨) | `f2ab7f28378e` |
| `rans_correction_nn.npz` | RANS 계수 보정(네트워크 다섯 개와 그 표준화) | `86286dcca4f0` |
| `op2_d500k_previous.npz` | 이전에 공개된 경계층 오퍼레이터(d500k; 배포된 체인에는 포함되지 않음) | `3c9e8d031fd4` |
| `airfrans_surface_tokens.npz` | 형상 잠재 벡터용 표면 점 토큰 | `cd27519b76dc` |
| `airfrans_train_stats.npz` | 유동장 디코더 출력 정규화 | `3da1cefe69d2` |

전체 해시는 `weights/SHA256SUMS`에 `sha256sum` 출력 그대로 들어 있으며, 이 파일에는 경로 없이 파일 이름만
나열되어 있습니다. 검증은 `(cd weights && shasum -a 256 -c SHA256SUMS)`로 하십시오.

## <a id="install"></a>설치

**애플리케이션과 `vortfoil` 명령(Windows, macOS, Linux).** Python도 함께 제공하는 [uv](https://docs.astral.sh/uv/)로 설치합니다:

1. uv를 한 번만 설치합니다:
   - Windows: `winget install --id=astral-sh.uv -e`
   - macOS: `brew install uv`
   - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. VortFoil을 설치합니다: `uv tool install --python 3.12 --managed-python vortfoil`. 이 저장소의 사본에서 설치하려면: `uv tool install --python 3.12 --managed-python ./VortFoil`. `--managed-python`은 uv가 자체 Python을 쓰도록 하며, 이 Python에는 창에 필요한 Tk가 들어 있습니다. Ubuntu의 Python 같은 시스템 Python에는 Tk가 없을 수 있습니다.
3. 터미널에서 `vortfoil`을 입력합니다. 애플리케이션 창이 열립니다([데스크톱 애플리케이션](#desktop-application) 참조).

같은 명령에는 다음 하위 명령이 있으며, 각 하위 명령은 `--help`를 받습니다:

- `vortfoil view`: 명령줄에서 하나의 창에 하나의 익형을 띄웁니다;
- `vortfoil predict`와 `vortfoil sweep`;
- `vortfoil about`과 `vortfoil selfcheck`.

`uv tool upgrade vortfoil`로 업데이트합니다. Linux에서는 창을 띄우려면 데스크톱 세션(디스플레이)이 필요합니다.

### 두 가지 명령: `vortfoil`과 `vortfoil_exp`

- `vortfoil`은 논문의 2차원 모델, 즉 한 단면에 대한 표면 모델(XFOIL 상태)과 RANS 유동장 디코더(RANS 상태)를
  애플리케이션 창과 위의 하위 명령에서 실행합니다. 이 창에는 3차원 날개가 없습니다(`vortfoil app --experimental`은 거부됩니다).
- `vortfoil_exp`는 같은 명령에 애플리케이션 창의 기능이 하나 더해진 것으로, 그 기능은 **실험적** 준3차원 날개입니다(단면 이름 옆의
  "3D wing" 상자). 이는 두 상태 중 어느 쪽의 VortFoil 단면이든 적용할 수 있는 고전적인 비선형 양력선(Phillips and Snyder, 2000)이며,
  크기와 관계없이 직선형에서 중간 정도의 후퇴각을 가진 평면형까지, 꺾임, 상반각, 그리고 NACA, UIUC 또는 파일 단면을 다룰 수 있습니다.
  논문에서 검증한 주장에는 포함되지 않습니다. `vortfoil_exp view`, `predict`, `sweep`, `about`, `selfcheck`는
  `vortfoil`의 것과 같습니다. 저장소 체크아웃에서는 `python code/vortfoil_app.py --experimental`입니다.
  `Section model` 행 아래의 회색 줄은 입력이 지켜야 할 조건을 알려 줍니다. 즉 모든 띠가 각 상태의 학습 레이놀즈 수 범위 안에 들도록 하는
  속도(이 속도는 띠의 시위에 따라 달라지며, 시위는 표, `Scale x`, `Strips`에서 정해집니다), 띠의 받음각 한계, 입력한 속도가 적합한지 여부,
  그리고 RANS 상태가 NACA 4/5자리 단면만으로 학습되었다는 점(단면이 그에 해당하지 않을 때에만 표시됩니다)입니다. UIUC 단면을
  날개에 적용하려면 `Airfoil` 상자에서 그 단면을 고르고(목록을 검색할 수 있습니다) `Geometry` 창에서
  **Selected row(s)** 또는 **All rows** 버튼을 누릅니다.

### <a id="uninstall"></a>제거

1. 프로그램을 제거합니다.
   - uv로 설치한 경우: `uv tool uninstall vortfoil`.
   - pip로 설치한 경우: `pip uninstall vortfoil`.
   - 체크아웃용으로 만든 환경: `conda env remove -n vortfoil`을 실행하거나 venv 폴더를 삭제한 다음, 저장소 폴더를 삭제합니다.
2. 애플리케이션이 보관하는 항목(UIUC 목록, 열어 본 단면, 로그)을 삭제합니다. Windows에서는 `%LOCALAPPDATA%\vortfoil`에, macOS와 Linux에서는 `~/.cache/vortfoil`에 있습니다.
3. `--shortcut`으로 만든 런처를 삭제합니다. Windows에서는 바탕 화면의 **VortFoil**, Linux에서는 `~/.local/share/applications/vortfoil.desktop`입니다(`--experimental`을 쓴 경우: **VortFoil (experimental)** 또는 `vortfoil-exp.desktop`). macOS에는 없습니다.
4. 다른 uv 도구가 사용하지 않는다면 uv의 Python과 다운로드 캐시를 제거합니다: `uv python uninstall 3.12`를 실행한 다음 `uv cache clean`을 실행합니다.
5. VortFoil 때문에만 uv를 설치했다면 마지막에, 즉 uv가 필요한 4단계 뒤에 제거합니다:
   - Windows: `winget uninstall --id=astral-sh.uv -e`
   - macOS: `brew uninstall uv`
   - Linux: `rm ~/.local/bin/uv ~/.local/bin/uvx`

**Python 라이브러리로 사용하는 경우**, 자신의 환경에 설치합니다:

```bash
pip install vortfoil            # PyPI에서
pip install -e .                # 이 저장소에서 설치(새 환경에서는 TensorFlow도 설치됩니다)
```

버전이 고정된 TensorFlow가 이미 있는 환경에서는 그것을 건드리지 않고 설치합니다:
`pip install --no-deps -e .`. Python 3.10에서 TensorFlow 2.10(CPU와 NVIDIA RTX
4060 Ti), 그리고 TensorFlow 2.18 / Keras 3.12로 검증했으며, 세 환경 모두 기록된 점수를 재현합니다.

```python
import vortfoil, numpy as np

vf = vortfoil.load_model()                       # mode="graph", device="cpu"
r  = vf.predict(cst=cst16, alpha=4.0, Re=1e6)    # 또는 coords=(P,2) 익형 좌표점
r.cl, r.cd, r.cm                                 # 배포된 양력, 항력, 모멘트
r.cl_gamma                                       # 순환 양력만, 2 * oint gamma ds
r.cp, r.stagnation, r.ue_te, r.theta_te, r.dstar_te
r.x_cp, r.theta, r.dstar                         # Cp의 가로 좌표; x/c = vortfoil.BL_X에서의 경계층 두께 [윗면, 아랫면]

vf.sweep(cst16, np.linspace(-4, 12, 33))         # 한 형상, 여러 받음각, 한 배치
vf.predict(coords=xy, alpha=np.linspace(-4, 12, 33), Re=1e6)   # 좌표로부터 같은 결과(적합은 한 번)
vf.gradients(cst=cst16, alpha=4.0, of="cl")      # 전체 체인을 통한 역방향 모드 d/dCST16
vortfoil.about()                                 # 가중치, 레시피, 모집단, 점수, 한계
```

CST 계수 대신 좌표를 주면, 형상을 먼저 16개 매개변수 형태로 적합하고 받음각은 적합된 시위선을 기준으로
측정하므로, 회전되거나 크기가 조정된 단면도 올바르게 처리됩니다. 적합 잔차와 회전은 `result.info`로 반환됩니다.

명령줄:

```bash
vortfoil about
vortfoil predict --cst 0.17,0.16,... --alpha 4 --Re 1e6
vortfoil sweep   --coords section.dat --alpha-range -4,12,17
vortfoil selfcheck --data data/eval/holdout_targets.npz     # 기록된 점수를 재현합니다
```

테스트: `VORTFOIL_EVAL_DATA=data/eval/holdout_targets.npz pytest -q tests/`.

### 입력 범위

표면 모델은 받음각 −10°에서 18°, 레이놀즈 수 5×10⁴에서 10⁷ 범위(비압축성, Ncrit 7, 닫힌 뒷전)의 XFOIL 해석 결과로
학습했고, −4°에서 12°, 2×10⁵에서 3×10⁶ 범위에서 평가했습니다. 학습 범위를 벗어나면 `predict()`, `sweep()`,
`gradients()`는 `vortfoil.RangeWarning`을 발생시키고 해당 행을 `result.info["outside_training_range"]`에 표시합니다.
좌표가 16개 매개변수 적합에서 시위의 0.5%보다 많이 벗어나거나 뒷전 간극이 시위의 1%를 넘을 때도 경고합니다.

### 임의 단면의 유동장

설치된 패키지에는 RANS 유동장 디코더(배포된 실행)가 들어 있으며, `vortfoil view`와 애플리케이션에서 이를 실행합니다.
저장소 체크아웃에서는 `code/field/predict_field.py`가 임의의 단면, 받음각, 레이놀즈 수에 대해 좌표만으로,
AirfRANS 파일도 XFOIL도 없이 이를 실행합니다:

```bash
python code/field/predict_field.py --coords section.dat --alpha 4 --Re 3e6 --out field.npz --png field.png
python code/field/predict_field.py --naca 23012 --alpha 8 --Re 4e6 --out field.npz
```

격자 위의 u, v, p, ν_t를 AirfRANS 단위(시위 1 m, 공기, U∞ = Re·ν)와 무차원화한 값(u/U∞, v/U∞, C_p,
ν_t/(U∞c))으로 기록하며, 사용한 오퍼레이터 상태(C_l^γ, s_sep)도 함께 기록합니다. 기본 격자는 바람 좌표계(자유류가 +x 방향)에서
x 방향으로 −0.5에서 1.5, y 방향으로 −0.5에서 0.5 시위의 범위를 0.005 시위 간격으로 덮습니다. `--points`는 사용자의 점을 받고,
`--frame body`는 시위 좌표계를 사용합니다.

배포된 디코더는 받음각 −5°에서 15°, Re 2×10⁶에서 6×10⁶의 NACA 4자리 및 5자리 단면에 대한 AirfRANS k–ω SST 유동장과
비-NACA UniFoil 익형 16,923개(Spalart–Allmaras, −2°에서 6°)를 함께 사용해 학습했습니다. AirfRANS 범위를 벗어나면
`FieldRangeWarning`을 발생시킵니다.

**다른 익형 계열: 정확도가 훨씬 낮습니다.** AirfRANS 프로토콜(k–ω SST, OpenFOAM v2112, 위의 범위 안)로 계산한 미관측 비-NACA UIUC 시험 단면 16개에서, 배포된 디코더의 평균 오차는 u_x, u_y, p, ν_t에서 각각 NACA 시험 수준의 14, 15, 46, 38배이며, 2.0.0의 AirfRANS 전용 디코더는 각각 27, 35, 68, 40배입니다(`results/cotrain_score_cmx_20261015T023434Z.json`). 아래의 단면별 수치는 그 AirfRANS 전용 디코더로 UIUC 단면 열 개에서 측정한 것입니다. 그 오차는 단면별로 해당 디코더의 NACA 시험 수준의 1.5배에서 80배였고, u_x는 평균 26배였습니다.

- 오차는 벽 근처에서 가장 크지만, 물체에서 떨어진 곳에서도 큽니다.
- NACA 형상에 가까운 단면은 시험 오차 분포의 꼬리 안에 머물렀습니다: RAE 2822 1.5배, NACA 64₂-415 2.8배, DU 84-132V 2.9배.
- 나머지는 크게 벗어났습니다: FX 63-137 7.7배, Clark Y 10배, NLF(1)-0416 13배, SD7003 15배, E387 26배, AH 79-100C 51배, S1223 80배.

형상 코드 거리 검사는 이들 중 어느 것도 걸러 내지 못했으므로, 익형 계열을 판별하는 장치가 아닙니다. 따라서 NACA 번호(`--naca`, 뷰어나 애플리케이션에서 고른 NACA 익형)로 주어지지 않은 모든 단면에는 그 사실을 알리는 `FieldRangeWarning`이 표시됩니다. 이런 유동장은 참고용으로만 읽으십시오. 표면 결과는 별도의 표면 모델에서 나오므로 영향을 받지 않습니다.

AirfRANS 시험 사례 200개에서 이 경로를 검증한 결과이며, 평가를 재현하는 대상인 2.0.0의 AirfRANS 전용 디코더로 실행했습니다
(`code/field/validate_predict_field.py`, `results/predict_field_validation.json`):

- **해당 디코더의 기록된 평가 입력:** 스크립트가 기록된 사례별 오차를 재현합니다.
- **사례의 441개 노드 단면 좌표만 사용:** 평균 오차는 기록된 오차 대비 u_x, u_y, p, ν_t에서 각각 0.99, 0.99, 1.07, 1.03배입니다.
- **성긴 161개 노드 단면(재샘플링하여 사용):** 대부분의 격자 노드가 경계층의 첫 번째 셀들 안에 있기 때문에 RANS 격자에서의 오차가
  벽 근처에서 커집니다(u_y 3.8배). 0.01 시위 격자에서는 성긴 입력과 조밀한 입력의 예측이 디코더 자체 시험 오차의 0.3%에서 6%만큼 다릅니다.

벽 근처 유동장이 중요하면 조밀한 좌표(최소 441개 점)를 주십시오.

### RANS 상태의 벽면 압력, 양력, 항력

이 디코더는 벽면 자체의 압력도 예측하므로(그곳에서 속도와 ν_t는 영이 됩니다), 유동장은 표면 모델의 XFOIL 상태 값과 나란히 자신의 기준 상태인 k–ω SST에서의 표면 값을 제공합니다. 디코더는 벽에 수직인 속도 기울기를 해상하지 못하며(첫 번째 격자 노드가 y+ ≈ 1에 있고, 2.0.0의 AirfRANS 전용 디코더에서 그 기울기로 계산한 표면 마찰은 RANS 표면 마찰의 2.3배입니다), 따라서 표면 마찰은 벽 함수로 구합니다. 벽에서 2×10⁻⁴ 시위 떨어진 로그층(시험 사례의 y+ 중앙값 31)에서의 디코더 접선 속도로부터 Spalding 법칙으로 마찰 속도를 구합니다. `predict_field.surface()`는 단면 노드에서의 벽면 C_p와 C_f, 그리고 양력과 항력을 반환합니다:

```python
import predict_field as PF          # code/field
w = PF.surface(PF.naca_coords("2412"), 4.0, 3e6, family="naca")
print(w["cl"], w["cd"], w["cd_pressure"], w["cd_friction"])
# w["x"], w["y"], w["cp"], w["cf"], w["upper"]: 단면 노드에서의 벽면 값(흐름이 뒷전(TE)을 향하는 곳에서 C_f > 0)
```

`predict_field.py`는 이 값들을 출력하고 결과 파일에 저장합니다(`wall_x`, `wall_y`, `wall_cp`, `wall_cf`, `wall_upper`). 뷰어와 애플리케이션은 이 C_p를 점선으로 표면 모델의 것 위에 겹쳐 그리고 그 양력, 항력, 모멘트를 나열하며(아래의 RANS 상태), CSV 내보내기에는 `_wall.csv`가 추가됩니다.

AirfRANS 시험 사례 200개의 RANS 벽면 값에 대해, 2.0.0의 AirfRANS 전용 디코더에서 단면 좌표만으로 비교한 결과
(`code/field/validate_field_surface.py`, `results/field_surface_validation.json`):

- **벽면 C_p:** 평균 RMSE 0.035, 시위의 5–95% 구간에서는 0.014입니다. 오차는 앞전 흡입 피크와 정체점에 집중됩니다. 받음각 구간별(−6–0°, 0–8°, 8–12°, 12–16°)로는 0.020, 0.018, 0.042, 0.120입니다.
- **양력:** RANS 양력 대비 R² 0.9993, 평균 상대 오차 0.029, Spearman 0.9997.
- **항력:** RANS 항력 대비 평균 상대 오차 0.14, Spearman 0.81. 표면 마찰 성분(벽 함수)의 평균 상대 오차는 0.045(Spearman 0.93)이고, 압력 성분이 더 약합니다(평균 상대 오차 0.38, Spearman 0.80).

배포된 디코더를 사용한 원고의 RANS 기준 평가(예측 유동장에 AirfRANS 힘 계산 루틴을 적용하고 같은 벽 함수를 사용;
`results/table4_row_cmx_s0.json`)에서는 양력의 평균 상대 오차가 0.054(Spearman 0.9995), 항력은
0.128(Spearman 0.743; 벽 기울기로 표면 마찰을 구하면 0.455)이며, 모멘트 MAE는 0.0023입니다.

이들은 SST 상태의 값입니다. 표면 모델의 계수는 XFOIL 상태의 값이며, 둘의 절대 수치를 서로 비교하지는 않습니다.

### RANS 상태의 경계층

`predict_field.boundary_layer()`는 디코더 자신의 유동에서 경계층을 읽어 냅니다. 각 표면에서 x/c = 0.01부터 0.99까지 단면 노드를 하나 걸러 하나씩 골라, 벽에 수직인 광선(길이 0.1 시위)을 따라 디코더의 속도와 압력을 샘플링하고, 이를 적분하여 운동량 두께 θ, 배제 두께 δ\*, 형상 계수 H, 가장자리 속도, 가장자리 거리를 구합니다. C_f는 위의 벽 함수에서 얻은 값입니다. 가장자리는 접선 속도의 와도 적분이 포화되는 지점입니다(이 기준은 경계층을 따라가며, 벽 기울기에 상대적인 문턱값은 대신 레이놀즈 수를 따라갑니다). 속도는 먼저 유동장 자체의 압력이 주는 비점성 속도로 나누므로, 경계층 앞쪽의 휘어진 유동이 가장자리를 움직이지 않습니다. 경계층이 읽히지 않는 스테이션은 NaN이며, 벽 근처 유동이 바깥 유동과 반대로 흐르는 스테이션에는 `separated` 표시가 붙습니다. AirfRANS 시험 사례 200개의 RANS 유동장에 같은 추출을 적용한 결과와 비교하면(2.0.0의 AirfRANS 전용 디코더에서 단면 좌표만 사용; `code/field/validate_field_boundary_layer.py`, `results/field_boundary_layer_validation.json`):

- **x/c 0.1–0.95의 부착 스테이션**(사전에 정한 한계, 즉 θ와 δ\*의 상대 오차 중앙값 0.20 이하를 충족함): 윗면 3.6%(θ)와 3.8%(δ\*), 아랫면 6.4%와 5.8%이고, H는 1.1%입니다. 90번째 백분위수는 윗면 19%와 19%, 아랫면 46%와 36%입니다.
- **아랫면은 사례별로 편차가 있습니다:** 오차 중앙값이 200개 사례 중 31개에서 20%를, 13개에서 50%를 넘습니다. 31개 중 21개는 8° 이상의 66개 사례에 속하며(그 범위의 얇은 단면 17개 중 9개, t/c < 9%), 윗면의 오차 중앙값은 5개 사례에서 20%를 넘고 50%를 넘는 사례는 없습니다.
- **뒷전(x/c > 0.95):** 1–3%. **앞전(x/c < 0.1): 검증되지 않음**, 윗면 10%, 아랫면 48%. 그림은 경계층을 x/c = 0.1부터 그리며, CSV 파일에는 0.01부터의 스테이션이 들어 있습니다.
- **박리 스테이션은 드뭅니다:** 38,421개 중 373개로, 32개 사례에 있으며 모두 윗면입니다. 그곳에서 θ와 δ\*는 (중앙값 기준) 2% 이내입니다.

RANS 상태의 경계층은 디코더 유동장의 판독값입니다. 그 정확도는 유동장이 벽 근처에서 얼마나 정확한지에 달려 있으며, 디코더의 학습 범위 밖에서는 경계층이 아예 읽히지 않는 경우가 많습니다.

### 하나의 익형, 하나의 창

`code/view_airfoil.py`는 익형(NACA 4자리 또는 5자리 번호나 `.dat` 파일), 받음각, 레이놀즈 수를 묻고, 모든 결과를 담은 하나의 창을 엽니다. 왼쪽에는 바람 좌표계에서의 네 개의 유동장 패널(RANS 유동장 디코더의 결과)이, 오른쪽에는 표면 압력, 두 표면의 경계층 두께, RANS 상태의 벽 함수 C_f, 그리고 모든 범위 경고를 담은 계수가 두 가지 상태로 표시됩니다:

- **XFOIL 상태**(실선): XFOIL로 학습한 표면 모델입니다. 와류 시트의 C_p, 그 δ\*와 θ, 그리고 C_l, C_d, C_m, 순환 양력, 정체점이 해당합니다. 학습 범위는 −10°에서 18°, Re 5×10⁴에서 10⁷입니다.
- **RANS 상태**(점선 C_p, 경계층은 점): AirfRANS k–ω SST 유동장으로 학습한 유동장 디코더 자체의 값입니다. 벽면 C_p, C_l, 압력 성분과 표면 마찰 성분을 가진 C_d, ¼ 시위점 기준 C_m, 벽 함수 C_f, 그리고 유동에서 읽어 낸 경계층(δ\*는 채운 점, θ는 빈 점)이며, 경계층은 검증된 x/c = 0.1부터 표시합니다(위 참조). 학습 범위는 −5°에서 15°, Re 2×10⁶에서 6×10⁶입니다.

각 상태는 자신의 범위를 가지며, 각자 자기 범위에 대해 경고합니다. 기본적으로 둘 다 그려집니다. 애플리케이션에서는 `Flow` 상자의 **XFOIL state**, **RANS state** 체크 상자로 표시할 상태를 고르며, 다시 계산하지 않고 다시 그리기만 합니다(둘 중 하나는 항상 체크된 상태로 남습니다). 학습 범위를 벗어난 상태도 계산은 되지만 흐리게 그려지고 "outside training range"(학습 범위 밖)라는 라벨이 붙으며 수치도 옅게 표시됩니다. 그것이 RANS 상태이면 네 개의 유동장 패널도 흐려지고, 범례, 상태 표시줄, 제목이 그 사실을 알립니다. matplotlib이 필요합니다. 창 크기를 바꾸면 그림 전체의 배율이 함께 바뀌며, 배치는 그려진 그대로 유지됩니다. 인수를 주면 질문을 건너뛰며, `--save`는 그림도 함께 저장합니다:

```bash
python code/view_airfoil.py
python code/view_airfoil.py --airfoil 23012 --alpha 6 --Re 3e6 --save view.png
```

![뷰어: 받음각 6도, Re 3e6에서의 NACA 2412, XFOIL 상태와 RANS 상태](docs/viewer_example.png)

### <a id="desktop-application"></a>데스크톱 애플리케이션

`code/vortfoil_app.py`는 애플리케이션 창으로 실행하는 VortFoil입니다. 단면은 세 가지 방법 중 하나로 고릅니다:

- NACA 4자리 또는 5자리 번호;
- UIUC Airfoil Coordinates Database 검색. 목록과 선택한 단면은 UIUC 사이트에서 내려받아 캐시에 저장하며, VortFoil은 이를 재배포하지 않습니다;
- 어떤 데이터베이스에도 없는 단면을 위한, 직접 만든 좌표 파일.

그다음 받음각, 레이놀즈 수, 유동장 상세도를 정하고 **Run** 버튼을 누릅니다. **Save figure** 버튼은 그림을 400 dpi(png, pdf 또는 svg)로 저장하고, **Export data (CSV)** 버튼은 결과의 모든 수치를 열을 설명하는 텍스트 파일과 함께 CSV 파일로 내보냅니다. 파일에는 계수(RANS 상태의 양력, 항력, 모멘트와, 각 상태가 학습 범위 안에 있는지를 나타내는 1/0 값 포함), 240개 패널에서의 표면 C_p, 121개 스테이션에서의 표면 모델 경계층, 모든 격자점의 유동장(U와 c 기준 단위와 SI 단위), RANS 상태의 벽면 값과 경계층(`_wall.csv`, `_boundary_layer_rans.csv`), 그리고 단면이 들어 있으며, `vortfoil view ... --csv DIR`도 같은 것을 씁니다. 애플리케이션은 세 가지 배치 중 그림이 가장 크게 보이는 것을 고릅니다:

- 나란히 배치(가로로 넓은 창);
- 표면 패널 한 줄 위에 유동장을 배치(보통 비율의 창);
- 모두 위아래로 쌓아 배치(세로로 긴 창). 세로로 세워 둔 화면에서는 컨트롤도 그림 위로 올라갑니다.

창 크기를 바꾸면 그림의 배율이 바뀝니다.

**설치.** [설치](#install)에서처럼 uv로 설치하면 `vortfoil`로 열 수 있습니다.

개발을 위해 대신 체크아웃에서 실행하려면, 애플리케이션에는 matplotlib과 Tk가 있는 Python 3.10–3.12가 필요합니다. Tk는 python.org 설치 프로그램, Anaconda, uv의 Python에 포함되어 있으며, Debian이나 Ubuntu에서는 `sudo apt install python3-tk`를 실행합니다. 저장소 폴더에서 설치하고 시작합니다:

```bash
conda create -n vortfoil python=3.11 -y && conda activate vortfoil      # 또는: python3 -m venv ~/vortfoil-env && source ~/vortfoil-env/bin/activate
pip install .
python code/vortfoil_app.py
```

`vortfoil app --shortcut`(또는 `python code/vortfoil_app.py --shortcut`)은 실행 중인 환경용 런처를 추가합니다:

- Windows에서는 바탕 화면에 **VortFoil**;
- Linux에서는 애플리케이션 메뉴의 항목;
- macOS에서는 터미널이 런처이므로(`vortfoil`) 아무것도 쓰지 않습니다.

`--experimental`을 주면(`vortfoil_exp app --shortcut`) 두 번째 런처 **VortFoil (experimental)**(Linux에서는 `vortfoil-exp.desktop`,
macOS에서는 `vortfoil_exp`)을 추가하며, 이 런처의 창에는 3차원 날개가 있습니다. 첫 번째 런처는 그대로 남습니다.

**참고.**
- 첫 실행에서는 모델을 불러오는 데 몇 초가 걸리며, 이후 실행은 보통 수준의 유동장 상세도에서 약 1–5 s가 걸립니다. 실행 한 번에 약 1.5 GB의 메모리가 필요합니다.
- UIUC 목록은 처음에는 인터넷 연결이 필요합니다.
- 목록과 열어 본 단면은 Windows에서는 `%LOCALAPPDATA%\vortfoil\uiuc`에, 그 외에서는 `~/.cache/vortfoil/uiuc`에 캐시됩니다.
- Windows 바로가기처럼 콘솔 없이 애플리케이션을 시작하면, 메시지는 그 폴더 옆의 `app.log`로 기록됩니다.
- 제거하려면 [제거](#uninstall)를 참고하십시오.
- `python code/vortfoil_app.py --selftest DIR`은 사람의 조작 없이 한 사례를 실행하고 그림과 창 캡처를 DIR에 저장하며, 설치를 빠르게 점검하는 용도입니다. `--experimental --selftest DIR --wing PRESET`은 날개에 대해 같은 일을 합니다.

## 실행 조건

아래에서 시간을 측정한 대상은 원고의 **계수 체인**(coefficient chain)입니다. 곧 시트 오퍼레이터와 그 패널 판독, 경계층 오퍼레이터,
그리고 고정된 두 닫힘식 모두입니다. 원고를 따라 ¼ 시위 모멘트와 정체점 위치 판독은 *제외합니다*. 따라서 이는
모멘트를 추가로 계산하고, 정체점 위치를 구하고, 모든 출력을 호스트로 복사하는 `predict()`의 비용이 아닙니다.

모델은 어느 구성에서나 같으며 비용만 달라집니다. 기록된 수치는 CPU의 즉시 실행(eager) 모드, 곧 가장 느린 구성에서 얻었습니다.
계수 체인의 사례당 비용(배치 1024): 즉시 실행 CPU 317 µs, 컴파일된 CPU 163 µs이며, RTX 4060 Ti에서는 컴파일된 체인이
배치 512에서 126 µs에 이르고 역방향 모드 형상 기울기는 약 395 µs입니다. 단일 질의의 비용은 즉시 실행 11.3 ms, 컴파일 1.30 ms입니다.
GPU에서의 즉시 실행은 CPU보다 훨씬 느리므로(단일 질의 93 ms) GPU에 배포하려면 컴파일해야 하며, 그래서 `mode="graph"`가 기본값입니다.

패널 영향 커널은 메모리에 `(batch, 240, 241)` 텐서를 실제로 만들어 내므로 GPU에서는 큰 배치가 메모리 제약을 받습니다.
`max_batch`는 작업을 나누어 처리하며 기본값은 GPU에서 512(8 GB 카드는 이를 넘으면 메모리가 부족합니다), CPU에서 4096입니다.
점수는 float32 반올림 수준에서 장치 간에 일치합니다. 2,737행 모집단에서 배포된 양력의 R² 값은 CPU에서 0.987952,
RTX 4060 Ti에서 0.987956입니다.

시간 측정 스크립트: `code/reproduce/vf_speed_surface.py`(기록된 구성)와 `code/reproduce/vf_speed_surface_modes.py`(모든 조건).
결과는 `results/_speed_modes_*.json`에 있습니다.

## 평가 모집단

이 저장소와 원고에는 다섯 가지 행 수가 나옵니다. 이들은 같은 홀드아웃 형상에 서로 다른 필터를 적용한 것이며,
서로 바꿔 쓸 수 없고, 한 모집단의 점수를 다른 모집단의 점수와 비교하는 일은 결코 없습니다.

| n | 이름 | 필터 | 수치가 있는 곳 |
|---|---|---|---|
| 2,678 | 부착 유동, 수렴 | 유효성 필터 **그리고** XFOIL의 전역 뉴턴 반복이 수렴 | `results/converged_set_numbers_FINAL.json`, `sets.attached_converged` |
| 2,737 | 부착 유동, 유효성 필터링 | 같은 격자에서 수렴 검사를 **적용하지 않음** | `results/converged_set_numbers_FINAL.json`, `sets.attached_all_gated` |
| 1,930 | 박리 유동 | 확장된 받음각, H_TE ≥ 6; 수렴하지 않은 행 없음 | `results/converged_set_numbers_FINAL.json`, `sets.separated` |
| 2,688 | 부착 유동, 스테이션 일치 BL | NeuralFoil의 32개 x/c 스테이션에서 경계층 참값을 가진 부착 유동 행 | `results/station_bl_compare.json` |
| 2,166 | 박리 유동, 스테이션 일치 BL | 박리 유동 집합에 같은 스테이션 일치를 적용 | `results/station_bl_compare_sep.json` |

원고는 부착 유동의 표면 및 항력 수치에 2,678행 모집단을 보고합니다. 제공되는 자체 점검(`vortfoil selfcheck`,
`code/smoke_confirmed_chain.py`, `tests/`)은 **2,737행** 모집단을 보고합니다. 제공되는 행별 기준 파일이 이 모집단을 대상으로
작성되었기 때문입니다. 이 점검은 패키지 자체의 기록된 점수를 재현할 뿐 원고의 부착 유동 표를 재현하는 것이 아니며,
2,678행 수치의 재현 명령도 아닙니다. 2,678행 수치는 이미 `results/converged_set_numbers_FINAL.json`에 계산되어 있으며,
그 파일을 쓴 스크립트는 `code/reproduce/vf_converged_set_numbers.py`이고, 스테이션 일치 경계층 비교는
`code/reproduce/station_bl_compare.py`에서 나옵니다.

## 결과 요약

**표면과 항력, 부착 유동 집합**(`data/eval/holdout_targets.npz`, 2,880행 격자를 유효성 + `Cd ∈ (1e-4, 0.1]` + `|Cl| ≤ 3` +
XFOIL 전역 뉴턴 반복 수렴으로 걸러 n = 2,678로 만든 것이며 참값은 Ncrit 7입니다. 수렴 검사로 제외된 59행에는
XFOIL 표면 해가 아예 없으므로 비교하는 모든 모델에 같은 행을 사용합니다. `results/converged_set_numbers_FINAL.json` 참조):

| 항목 | VortFoil | NeuralFoil xlarge | 출처 |
|---|---|---|---|
| Cl R²(배포: 순환 + 양력 닫힘식) | 0.989 | 0.995 | `results/converged_set_numbers_FINAL.json` (`sets.attached_converged.bands.all.cl_deployed`, `cl_nf`) |
| Cl R²(순환 판독만, Cl^γ) | 0.987 | — | `results/converged_set_numbers_FINAL.json` (`...cl_gamma`) |
| Cd R²(배포된 닫힘식, `fitted_poly2`) | 0.920 | 0.955 | `results/converged_set_numbers_FINAL.json` (`...cd`, `cd_nf`) |
| Cm R²(압력 적분 판독) | 0.953 | 0.996 | `results/converged_set_numbers_FINAL.json` (`...cm_press`, `cm_nf`) |

**박리 유동 홀드아웃**(같은 80개 홀드아웃 UIUC 형상, α ∈ {−10,−8,−6,13…18}, n = 1,930):

| 항목 | VortFoil | NeuralFoil xlarge | 출처 |
|---|---|---|---|
| Cl R², 전체 행(배포된 양력; 순환만 쓰면 0.989) | 0.994 | 0.999 | `results/converged_set_numbers_FINAL.json` (`sets.separated.bands.all.cl_deployed`, `cl_nf`) |
| Cd R², 전체 행 | 0.934 | 0.983 | `results/converged_set_numbers_FINAL.json` (`...cd`, `cd_nf`) |
| Cm R², 전체 행(압력 적분 판독) | 0.843 | 0.994 | `results/converged_set_numbers_FINAL.json` (`...cm_press`, `cm_nf`) |
| Cd R², 후류 전진 적분(기준선), H_TE ≥ 10 (n = 462) | −0.276 | — | `results/FINDING_closure_track.json`, 발견 F11 |

R² 값이 음수인 마지막 행이 바로 전진 적분 방식 대신 적합된 닫힘식(`fitted_poly2`)이 배포 기본값인 이유입니다.
닫힘식 선택 과정의 전체 경위는 `results/FINDING_closure_track.json`(발견 F3–F16)을 참조하십시오.

**배포된 유동장 디코더**(모든 격자 노드에 대한 채널별 표준화 MSE, 사례마다 같은 가중치를 둔 평균입니다. σ는 200개 사례의
희소 태스크 학습 분할에서 뽑은 50개 사례의 모든 노드로 구했습니다. 공표된 값은 해당 태스크 학습 분할의 모든 노드로 σ를
구하므로, 이 값은 그에 비해 보수적입니다. `results/cotrain_score_cmx_20261015T023434Z.json`):

| 집합 | ux | uy | p | nut |
|---|---|---|---|---|
| AirfRANS `full_test`, n = 200 (k–ω SST) | 6.52e-4 | 9.69e-4 | 1.52e-3 | 3.52e-3 |
| 학습에서 보지 않은 비-NACA UIUC 단면 16개(k–ω SST, AirfRANS 생성기) | 9.13e-3 | 1.50e-2 | 7.04e-2 | 1.34e-1 |
| 홀드아웃 UniFoil 익형 150개(Spalart–Allmaras; 저장된 행의 부분 표본이며 위의 행들과 비교할 수 없음) | 3.00e-4 | 9.27e-5 | 2.20e-4 | 5.45e-5 |

같은 수의 최적화 스텝으로 학습한 AirfRANS 전용 디코더 대비 쌍체 사례 부트스트랩 오차비는 AirfRANS에서 1.00, 1.36,
0.57, 0.51이며(NACA 속도 오차의 개선은 학습 길이에서 옵니다), 비-NACA 단면 16개에서는 0.39, 0.30, 0.44, 0.50으로
모두 1보다 낮습니다(`results/stepmatch_score_20261015T070933Z.json`).

**RANS 계수 보정**(`results/rans_correction_nn_20261015.json`; 내보낸 가중치는 적합의 예측을 2e-16 이내로 재현합니다,
`results/rans_correction_nn_export_check.json`): AirfRANS 시험 분할(n = 200)에서 RANS 대비 표면 브랜치의 양력 / 항력
평균 상대 오차는 7.7% / 25.9%에서 2.9% / 4.3%로, 모멘트 MAE는 0.0082에서 0.0028로 줄어듭니다. 학습에서 보지 않은
비-NACA SST 단면 107개에서는 양력 MAE가 0.037에서 0.028로, 항력 오차가 30.2%에서 8.9%로, 모멘트 MAE가
0.0096에서 0.0071로 줄어듭니다. 보정된 계수와 유동장 디코더의 적분 계수가 일치하도록 제약하지는 않았으며, 시험 분할에서
두 값의 차이는 양력 6.8%, 항력 14.1%입니다(`results/consistency_field_vs_corrected_20261029.json`).

**AirfRANS 전용 유동장 디코더**(원고의 제거 실험용 디코더이며 릴리스 2.0.0은 오퍼레이터 조건부 디코더를 배포했습니다.
AirfRANS `full_test`, n = 200, 같은 지표):

| 디코더(선택된 실행) | ux | uy | p | nut | 출처 |
|---|---|---|---|---|---|
| 오퍼레이터 조건부(`field_decoder_f0cm_s1`, 실행 1) | 1.26e-3 | 1.27e-3 | 2.13e-3 | 7.06e-3 | `results/f0cm_final_rescore_20261002T051450Z/result.json` (`mean.CM`) |
| XFOIL 조건부(`field_decoder_f0a_s2`, 실행 2) | 1.61e-3 | 2.07e-3 | 3.54e-3 | 9.34e-3 | 같은 파일(`mean.XF2`); `evaluation_runs/_nf_shape_f0a_s2__*` |
| MARIO, 공표된 집계값 | 1.52e-3 | 1.13e-3 | 2.40e-3 | 9.60e-3 | Catalani et al. 2025 |

두 디코더는 같은 AirfRANS 사례 780개로 학습했습니다. XFOIL 조건부 디코더 대비 오퍼레이터 조건부 디코더의 쌍체 사례 부트스트랩 오차비
(95% 신뢰구간; 200개 중 개선된 사례 수)는 ux, uy, p에서는 두 디코더의 오차가 같음을 뜻하는 값보다 낮고, nut에서는 그 값을 포함합니다:
0.78 [0.64, 0.91] (131), 0.62 [0.46, 0.81] (133), 0.60 [0.28, 0.96] (114), 0.76 [0.33, 1.63] (155)
(같은 파일의 `paired_vs_xfoil_conditioned_run2.CM`). 오퍼레이터 조건부 디코더의 오차는 MARIO가 공표한 오차보다
ux, p, nut에서는 낮고(0.83×, 0.89×, 0.74×) uy에서는 높습니다(1.13×)(`vs_MARIO.CM`). MARIO는 이 네 개의 집계값만 공표합니다.

- **드롭인(drop-in).** 재학습 없이 오퍼레이터가 공급하는 입력을 받은 XFOIL 조건부 디코더는 모든 채널에서 더 나쁩니다:
  2.38 [1.84, 3.14] / 3.79 [2.63, 5.57] / 2.97 [1.54, 7.50] / 3.41 [2.02, 6.80]. 양력 입력만 바꾸면
  1.21 / 1.58 / 0.99 / 1.12입니다(`paired_vs_xfoil_conditioned_run2.DB`, `.DL`).
- **양력 입력 대체.** 추론 시 XFOIL의 C_l을 쓰면 오퍼레이터 조건부 디코더의 오차가 커지며(ux 1.44, uy 1.71),
  편향을 보정한 C_l로도 일부만 회복됩니다(ux 1.08)(`cl_substitution_vs_CM`).
- **박리 규칙 민감도.** 평가 시 두 뒷전 형상 계수의 최댓값 대신 평균을 넣으면
  1.05 [1.02, 1.10] / 1.13 [1.08, 1.19] / 1.14 [1.03, 1.55] / 1.01 [0.89, 1.14]입니다(`eval_rule_sensitivity_CMm_over_CM`).
- **받음각 구간**(`results/band_table_f0cm_s1.json`, n = 55 / 79 / 44 / 22): XFOIL 조건부 디코더 대비 개선은
  [12, 16)°에서 가장 크며 모든 채널에서 0.41–0.70×입니다. MARIO의 통합 집계값과 비교하면 [−6, 0)°와 [0, 8)° 구간은
  모든 채널에서 MARIO보다 낮고, [8, 12)°는 ux(1.23×)와 uy(1.60×)에서 MARIO보다 높으며, [12, 16)°는 네 채널 모두에서
  MARIO보다 높습니다(1.68× / 3.73× / 5.45× / 4.89×).
- **앞 단계 입력.** 페인트의 순환 스케일을 정하는 시트의 순환 양력은 200개 시험 사례에서 XFOIL의 강제 천이 양력 대비
  R² 0.970을 기록합니다(`results/step2_eval_f0cm_20260925T213247Z/result.json`, `provider_vs_xfoil_diagnostic.cl_R2`).

`figures/vf_field_f0cm_tp_pos.{pdf,png}`(저장된 수치는 `figures/vf_field_f0cm_tp_pos_numbers.json`)은
시험 사례 `airFoil2D_SST_39.31_8.883_4.707_0.0_12.552`(U∞ = 39.3 m/s, α = 8.88°, 두께 12.6%인 대칭 NACA 네 자리 단면)에서
AirfRANS 전용 오퍼레이터 조건부 디코더(2.0.0에서 배포)를 RANS 참값과 나란히 그려 보여 줍니다. 이 사례는 그림을 그리기 전에 정한 규칙으로
골랐습니다. α ≥ 8°인 시험 사례 66개 중에서 디코더의 표준화된 전체 격자 MSE 네 개의 평균이 가장 낮은 사례이며(실행 1,
`results/step2_eval_f0cm_20260925T213247Z/per_case.npz`, `mse_all_*_s1`), 전체 200개 중 14위입니다. 각 채널은 RANS 열과
디코더 열이 함께 쓰는 자기 색 눈금을 가집니다. u∥(자유류 방향)는 자유류 속도를 가운데로 둔 coolwarm으로, 빠르면 빨강,
느리면 파랑입니다. u⊥는 영점을 가운데로 둔 coolwarm, p/ρ는 ±U∞²/2(C_p = ±1)로 범위를 제한한 RdYlBu_r여서 흡입 정점은
포화됩니다. ν_t는 로그 눈금의 YlGnBu입니다.

### 추론 비용

`results/_speed_confirmed_summary.json`(생성 스크립트: `code/reproduce/vf_speed_surface.py`, `vf_speed_nf.py`, `sepeval/xfoil_time_win.py`)에는 배치 크기 1–2737에서의 표면 체인 사례당 비용(시트 오퍼레이터 + Cl^γ 판독 9.95 µs; 계수 체인은 시트와 경계층 순전파에 두 고정 닫힘식을 더한 것으로 모멘트와 정체점 위치 판독을 제외하며, 노트북 CPU, TF 2.10 즉시 실행에서 B = 1024일 때 사례당 318 µs; 역방향 모드 형상 기울기는 Cl^γ에 대해 시트 순전파의 2.34×, 배포된 Cl은 계수 체인의 3.13×, Cd는 3.12×이며, 둘 다 경계층 오퍼레이터의 역전파를 공유하고 `results/_grad_check_*.json`에서 중심 유한 차분과 대조해 검증했습니다), 같은 행에 대한 NeuralFoil xlarge(배치 처리 사례당 28.8 µs, 단일 질의 0.691 ms), XFOIL 단일 프로세스(사례당 중앙값 0.104 s, Xeon), 그리고 장치가 다른 참고값으로서 MARIO가 공표한 88 ms/sample(A100)이 들어 있습니다. 절대값은 호스트 부하에 좌우되며, 같은 실행 안에서의 비율이 안정적인 통계량입니다.

유동장 경로: AirfRANS 전용 오퍼레이터 조건부 디코더의 구성 요소별 마이크로벤치마크(배포된 디코더도 구조와 입력은 같습니다; `results/_speed_confirmed_field_f0cm_20261002T053715Z.json`, 생성 스크립트 `code/reproduce/vf_speed_field.py`; NumPy 인코더와 디코더, 각 시험 사례를 자신의 RANS 격자 노드에서 질의, 세 사례에서 사례당 점 수의 중앙값 172,029개)입니다. 사례당 조건 입력 0.0131 s, 물리 페인트 5.38 s, 최근접 패널과 중복 거리 탐색 2.90 s, 형상 인코더 0.0000876 s, 디코더 1.29 s(10³개 점당 7.35 ms)입니다. 이는 구성 요소별 시간이며 종단 간 지연 시간이 아닙니다. MARIO의 0.088 s는 하드웨어와 시간 측정 경계가 다른 공표값이므로 속도비는 계산하지 않습니다.

구현 조건(`code/reproduce/vf_speed_surface_modes.py`, `results/_speed_modes_*.json`): 배포된 즉시 실행 CPU 수치는 상한입니다. 같은 계수 체인을 `tf.function`으로 컴파일하면(XLA 없음) 노트북 CPU에서 B = 1024일 때 163 µs/case(즉시 실행은 317), 단일 질의는 1.30 ms(11.3 ms)가 되고, RTX 4060 Ti(8 GB, Windows)에서는 컴파일된 체인이 B = 512에서 126 µs/case에 이르며 기울기는 ~395 µs/case입니다. 반면 GPU의 즉시 실행은 훨씬 느리며(단일 질의 93 ms), 이 구현에서는 배치 ≥ 1024가 카드의 메모리를 모두 소진합니다.

## 재현

두 개의 스모크 테스트가 제공되는 가중치를 봉인된 행별 수치와 대조합니다. 이 테스트는 수렴 검사를 *적용하지 않은* 유효성 필터링 행(n = 2,737)에서
평가합니다(위의 *평가 모집단* 참조). 봉인된 기준 파일이 이 모집단에서 작성되었으므로, 목표값은 위에 보고한 수렴 집합 값(0.987과 0.920)이 아니라
전체 행 값(Cl^γ 0.9862, Cd 0.8904)입니다. 이는 가중치 무결성 점검이며 원고의 부착 유동 표를 재현하는 것이 아닙니다.
`results/converged_set_numbers_FINAL.json`에는 두 모집단이 나란히 들어 있습니다.

```bash
# TF 2.10 환경 — 표면/항력 수치를 비트 단위로 그대로 재현합니다
python code/smoke_confirmed_chain.py

# `airfrans` 패키지가 있는 TF 2.18 환경 — AirfRANS 전용 유동장 디코더(2.0.0에서 배포)의 순전파
python code/smoke_field_decoder.py
```

보고된 수치의 근거가 되는 스크립트이며, 저장소 루트에서 실행합니다:

```bash
# 부착 유동 홀드아웃에서의 표면 계약(Cl/Cm/쿠타 판독)
python code/reproduce/surface_contract_cst16.py <out.npz> <out.json>

# 홀드아웃 스키마를 따르는 임의의 타깃 npz에 대한 물리 항력 경로 + NeuralFoil
python code/reproduce/eval_on_targets.py data/eval/holdout_targets.npz <out.npz> [op2_weights]

# AirfRANS full_test에서의 AirfRANS 전용 디코더 유동장 수치(외부 데이터셋 필요)
export AIRFRANS_ROOT=/path/to/AirfRANS/Dataset
python code/field/f0cm_final_rescore.py --workers 8    # 유동장 오차, 구간, 쌍체 비율, 드롭인, C_l 대체, s 규칙 민감도
python code/field/f0cm_final_rescore.py --limit 2      # 빠른 점검: 제공된 기록의 0-1행, 아무것도 쓰지 않음
python code/field/step2_eval_f0c.py                    # 오퍼레이터 조건부 실행 세 개(results/step2_eval_f0cm_*)
python code/field/validate_predict_field.py            # AirfRANS 전용 디코더의 평가에 대한 predict_field.py 검증(results/predict_field_validation.json)
python code/field/validate_field_surface.py            # RANS 벽면 값에 대한 predict_field.surface() 검증(results/field_surface_validation.json)

# AirfRANS 전용 디코더의 유동장 그림
python code/field/step2_store_preds.py --weights-tag f0cm --seed 1 --name airFoil2D_SST_39.31_8.883_4.707_0.0_12.552
python code/figures/vf_field_f2_contour.py --preds results/preds_f0cm_s1 --tag vf_field_f0cm_tp --layout truth_pred \
    --frame wind --label "operator-conditioned decoder" --case airFoil2D_SST_39.31_8.883_4.707_0.0_12.552 \
    --case-tag pos --p-qinf --print

# 유동장 시간 측정
python code/reproduce/vf_speed_field.py

# 임의 단면의 RANS 보정 계수(배포된 보정)
vortfoil predict --coords section.dat --alpha 4 --Re 3e6 --rans
```

실행을 고른 검증 점수는 `code/field/field_holdout_score.py`에서 나오며, 이 스크립트에는 학습 코퍼스가
필요합니다(제공되지 않음; `VORTFOIL_CORPUS_DIR`).

`code/README_code.md`에는 모든 스크립트와 이들이 따르는 `VORTFOIL_*` / `AIRFRANS_ROOT` 환경 변수가 나열되어 있습니다.

## 환경

- **표면 체인(OP1 + OP2 + 닫힘식):** TensorFlow 2.10, NumPy. 봉인된 표면/항력 수치가 산출된 바로 그 환경입니다.
- **유동장 체인:** TensorFlow 2.18과 `airfrans` 패키지(`manifest.json`과 점 로딩용), 그리고 `pyvista`(`airfrans`의 의존 패키지).
- 환경 간 편차는 작지만 분명히 존재합니다. TF 2.18에서는 봉인된 Cd R²가 TF 2.10 값에서 5e-7만큼 달라집니다. 봉인된 자릿수는
  TF 2.10으로 재현하십시오. 쫓아야 할 수치상의 불일치가 아닙니다.

## 데이터

- **AirfRANS**(Bonnet et al., 2022; ODbL v1.0)는 외부 데이터셋이며 여기서 재배포하지 않습니다. 단 하나의 예외는
  `weights/airfrans_surface_tokens.npz`로, 800개 사례의 표면 점과 이름을 담은 AirfRANS 발췌본이며 ODbL에 따라 제공됩니다.
  유동장 디코더와 유동장 그림은 AirfRANS로부터 만들어집니다. 출처 표시, 고지 사항, 전체 파일 목록은
  [`DATA_LICENSES.md`](DATA_LICENSES.md)에 있습니다. AirfRANS를 다루는 모든 작업에는 `AIRFRANS_ROOT`를 로컬 `Dataset/`
  디렉터리로 설정하십시오(`code/README_code.md` 참조).
- **UniFoil**(Kanchi et al., 2025; Harvard Dataverse, CC0 1.0)은 외부 데이터셋이며 여기서 재배포하지 않습니다. UniFoil
  익형을 사용해 학습한 디코더(`weights/field_decoder_cmx_s*.npz`, `field_decoder_cmd_s*.npz`)는 그 출처 표시와 함께 이 저장소의
  라이선스로 제공합니다(`DATA_LICENSES.md`).
- **`data/eval/`** 디렉터리에는 이 README의 모든 표면 수치를 만드는 데 쓴 XFOIL 평가 타깃이 있습니다. `holdout_targets.npz`(부착 유동 홀드아웃,
  2,880행 격자를 2,737행으로 걸러낸 것), `attached_surface_targets_v2.npz`와 `sep_eval_targets_v2.npz`(부착 유동과 박리 유동의
  표면/경계층 타깃이며, 후자는 NeuralFoil의 32개 스테이션에서의 경계층 상태를 담고 있습니다), 그리고 `sep_eval_coords.npz`(박리 유동 집합의
  질의 좌표)입니다.
- 학습 코퍼스(XFOIL 경계층 데크, AirfRANS 유동장 학습용 npz 파일)는 제공되지 않습니다. 각각 0.5–3.1 GB이고, 유동장 코퍼스는
  AirfRANS의 파생물이기 때문입니다. 오퍼레이터 조건부 디코더는 `_nf_train_v5_full800_provider_maxs.npz`(sha256
  `a387d918…`, 각 `field_decoder_f0cm_s*.npz`의 `run_config`에 명시됨)로 학습했으며, 이 파일도 제공되지 않습니다. 배포된
  디코더는 `_nf_unifoil_train_full_thin3_provider.npz`(sha256 `1e6ed4cf…`, 각 `field_decoder_cmx_s*.npz`의 `run_config`에
  명시됨)를 추가로 사용하며, 이 파일도 제공되지 않습니다.
  `code/README_code.md`에 각 코퍼스를 만드는 스크립트가 나와 있습니다.

## 수치의 출처

이 README의 모든 수치는 저자들의 연구 저장소에 있는 봉인된 기록의 사본입니다. `results/*.json`은 그 기록을 그대로 복사한 것이며
정본입니다. 이 README의 반올림 값이 아니라 위의 표에 적힌 키 경로를 인용하십시오. 이 중 여섯 개는 일부를 덜어낸 사본으로,
각각 출처와 생략한 내용을 밝히는 `note` 또는 `release_note`가 있습니다(`field_deployed_weights.json`,
`FINDING_closure_track.json`, `_speed_confirmed_summary.json`, `step2_eval_f0cm_20260925T213247Z/result.json`,
`one_rule_retrain/f0cm_summary.json`, `operator_conditioned_corpus_provenance.json`). `f0cm_holdout_rescore.json`은
검증 기록의 한 블록을 발췌한 것입니다. `operator_conditioned_corpus_provenance.json`은 오퍼레이터 조건부 디코더 학습 코퍼스(sha256
`a387d918…`, 800개 사례에서 나온 12,009,281행)의 생성 기록입니다. 이 파일은 XFOIL 입력으로 코퍼스를 만들면 XFOIL 조건부 디코더에 저장된
페인트 채널이 비트 단위로 똑같이 재현된다는 원고의 서술을 뒷받침하며, 이 점검은 800개 사례 중 세 개를 대상으로 합니다.
`REFACTOR_MANIFEST_v2.json`에는 복사된 파일의 출처와 sha256이 기록되어 있습니다. `figures/`에는 원고 그림과 함께, 그림 출처 규율이 요구하는
`*_numbers.json` 파일이 있습니다. 각 그림의 수치는 그려진 플롯에서 읽은 것이 아니라 그 생성 스크립트 옆에 저장됩니다.
제공되는 그림은 생성 스크립트의 `--print` 렌더링이며, 원고의 본문 폭(5.40 in)으로 그려져 원고에 원래 크기 그대로 삽입됩니다.

## 라이선스

GNU General Public License v3.0 또는 그 이후 버전 — [`LICENSE`](LICENSE)를 참고하십시오. `xfoil_closures.py`는 GPL v2 또는 그 이후 버전으로
라이선스된 XFOIL 6.99(Drela와 Youngren)를 한 줄씩 옮긴 번역이며, 그 출처와 v3를 택한 이유는 [`NOTICE`](NOTICE)에 있습니다.
데이터 라이선스와 출처 표시(AirfRANS, UniFoil과 UIUC 좌표)는
[`DATA_LICENSES.md`](DATA_LICENSES.md)에 있습니다.

## 인용

[`CITATION.cff`](CITATION.cff)를 참고하십시오. 소프트웨어 DOI: [10.5281/zenodo.23336871](https://doi.org/10.5281/zenodo.23336871).
