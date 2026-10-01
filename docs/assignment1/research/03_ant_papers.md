# Ant 로봇에 적용된 논문 목록 (검증된 것만)

작성일: 2026-09-30
대상 과제: IsaacLab `Isaac-Ant-v0`, PPO, 처음 보는 지형 유형에서 추가 학습 없이(zero-shot) 걷기

## 읽는 법

- **검증 방법**: 모든 논문은 arXiv PDF 전문을 내려받아 본문에서 "Ant/ant"가 나오는 부분을 직접 확인했습니다. 게재처는 PDF 머리말, 학회 논문집(PMLR, NeurIPS proceedings) 또는 학회 공식 페이지로 확인했습니다.
- **수치**: 본문이나 표에 **글자로 적힌 숫자만** 옮겼습니다. 결과가 그래프로만 나온 경우에는 "그래프만 있음"이라고 적고 숫자는 넣지 않았습니다(그래프에서 눈으로 읽은 값은 검증된 수치가 아니기 때문).
- **동료 평가 아님**: arXiv에만 올라온 논문은 "arXiv만, 동료 평가 아님"으로 표시했습니다.
- **추론**: 우리 과제에 도움이 될지에 대한 판단은 논문에 적힌 내용이 아니라 저의 해석이므로 모두 **[추론]** 으로 표시했습니다.
- **Ant 버전 표기**: Gym Ant = OpenAI Gym/MuJoCo Ant(다리 4개, 구동 관절 8개). "DeepMind Quadruped(8구동)"은 Heess 2017에 나오는, 구동 관절이 8개인 Ant 비슷한 몸체입니다.

---

## 1. 전체 표

| # | 제목 | 저자/연도 | 게재처 (동료 평가) | Ant 버전 | 적용한 방법 | Ant에서 보고된 결과 | 관련성 |
|---|---|---|---|---|---|---|---|
| 1 | Emergence of Locomotion Behaviours in Rich Environments | Heess 외, 2017 | **arXiv만, 동료 평가 아님** | DeepMind Quadruped (12 DoF, 구동 관절 8개) | 여러 가지 지형을 절차적으로 생성, 분산 PPO(DPPO), 지형 높이 정보 관측 | 다양한 지형, 벽 사이 지그재그 통과(slalom), 틈(gap), 허들 코스를 "상당히 안정적으로" 통과함. **허들에서 학습한 정책이 평지에서 학습한 정책보다** 처음 보는 마찰, 울퉁불퉁한 표면(rubble), 구동기 힘 변화, 경사에서 더 강건한 경향을 보임(Fig 6b). 정규화한 그래프만 있고 숫자는 없음 | **높음** |
| 2 | Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning | Makoviychuk 외, 2021 | NeurIPS 2021 Datasets & Benchmarks (동료 평가) | Isaac Gym Ant (평지) | GPU 대량 병렬 PPO | A100 GPU 1장에서 **20초 만에 보상 3000**(쓸 만한 보행) 도달. 병렬 환경을 256개에서 8192개로 늘리면 보상 7000 도달 시간이 **약 1000초에서 약 100초로** 줄어듦. 가장 좋은 설정은 환경 8192개, horizon 16 | 중간 |
| 3 | Brax – A Differentiable Physics Engine for Large Scale Rigid Body Simulation | Freeman 외, 2021 | NeurIPS 2021 Datasets & Benchmarks (동료 평가) | Brax Ant (관측 87차원, 행동 8차원) | 가속기(TPU/GPU)용 JAX PPO | Brax Ant는 "10초 정도"에 쓸 만한 보행에 도달하고, 일반적인 PPO + MuJoCo-Ant-v2는 "30분 가까이" 걸림(둘 다 1천만 스텝, 본문 설명). 지형 실험은 없음 | 낮음 |
| 4 | Contextualize Me – The Case for Context in RL (CARL) | Benjamins 외, 2023 | TMLR 06/2023 (동료 평가) | Brax Ant (`CARLAnt`) | 물리 조건("context")을 바꿀 수 있는 벤치마크 | CARLAnt에서 바꿀 수 있는 값: 구동기 힘(기본 300), 마찰(0.6), 중력(-9.8), 관절 강성·감쇠, 몸통 질량(10). **Ant 자체의 실험 결과는 본문에서 찾지 못함**(실험은 Pendulum 등으로 함) | 중간 (도구로) |
| 5 | Max-Min Off-Policy Actor-Critic … Worst-Case Robustness (M2TD3) | Tanabe 외, 2022 | NeurIPS 2022 (동료 평가) | Gym Ant | 최악 조건 최적화(max-min), DR·RARL과 비교 | 가장 나쁜 조건에서의 점수(×10³, 10번 실행): Ant1(몸통 질량 0.1~3.0배) **M2TD3 3.84 / DR 3.51 / RARL −1.24**. Ant2(+앞왼다리 질량) **4.13 / 1.64 / −1.77**. Ant3(+앞오른다리 질량) **0.10 / −0.32 / −2.38**. RARL은 Ant1에서 평균 성능조차 제대로 학습하지 못했다고 보고 | 중간 |
| 6 | Action Robust RL and Applications in Continuous Control | Tessler 외, 2019 | ICML 2019, PMLR 97 (동료 평가) | Gym Ant | 행동에 섞는 적대적 교란(PR/NR-MDP) | Ant는 부록 Fig 12(행동 노이즈에 대한 강건성)에만 나옴. 그래프만 있음 | 낮음~중간 |
| 7 | RL with Adaptive Curriculum Dynamics Randomization for Fault-Tolerant Robot Control (ACDR) | Okamoto 외, 2021 | **arXiv만, 동료 평가 아님** (학회 게재를 확인하지 못함) | Gym Ant-v2 | PPO + 구동기 고장을 무작위로 주는 DR + 적응형 커리큘럼 | 쉬운 것부터(easy2hard)보다 **어려운 것부터(hard2easy) 커리큘럼이 더 효과적**이라고 보고. 평균 보상과 걸은 거리에서 기존 방법보다 좋다고 함. 그래프만 있음 | 중간 |
| 8 | Model-Agnostic Meta-Learning (MAML) | Finn 외, 2017 | ICML 2017, PMLR 70 (동료 평가) | Gym/MuJoCo Ant | 메타러닝(목표 속도 0~3.0, 앞/뒤 방향) | 1~3번의 경사 업데이트만으로 새 속도·방향에 적응. 무작위 초기화 기준선은 Ant에서 보상 < −25. 그래프만 있음 | 낮음 |
| 9 | Learning to Adapt in Dynamic, Real-World Environments through Meta-RL (GrBAL/ReBAL) | Nagabandi, Clavera 외, 2019 | ICLR 2019 (동료 평가) | Gym Ant | 모델 기반 메타 RL로 온라인 적응 | **"Ant 다리 고장"**: 학습 때 다리 하나를 무작위로 고장 냄. 테스트에서는 **학습 때 고장 낸 적 없는 다리**를 고장 냄. 빠른 적응 실험에서 모델 기반 oracle보다 좋음. 최종 성능은 모델 프리 메타 RL보다 낮지만 데이터는 약 1000배 적게 씀. 그래프만 있음. (경사 지형 실험은 HalfCheetah로 했고 Ant가 아님) | 중간 |
| 10 | PEARL: Efficient Off-Policy Meta-RL via Probabilistic Context Variables | Rakelly 외, 2019 | ICML 2019, PMLR 97 (동료 평가) | Gym Ant | 잠재 컨텍스트 추론형 메타 RL | Ant-Fwd-Back(과제 2개), Ant-Goal-2D(학습 100개 / 테스트 30개 목표). 초록에는 "샘플 효율 20~100배"라고 나오지만 전체 과제를 합친 주장임. Ant 수치는 그래프만 있음 | 낮음 |
| 11 | Data-Efficient Hierarchical RL (HIRO) | Nachum 외, 2018 | NeurIPS 2018 (동료 평가) | Gym Ant (Gather/Maze/Push/Fall) | 계층형 RL(목표를 주는 상위 정책 + 하위 정책), off-policy 보정 | 1천만 스텝, 10개 시드(Table 1): **Ant Gather 3.02±1.49, Ant Maze 성공률 0.99±0.01, Ant Push 0.92±0.04, Ant Fall 0.66±0.07**. 비교한 SNN4HRL: 1.92 / 0.0 / 0.02 / 0.0 | 낮음 |
| 12 | Stochastic Neural Networks for HRL (SNN4HRL) | Florensa 외, 2017 | ICLR 2017 (동료 평가) | Gym(rllab) Ant | 사전학습한 기술 + 상위 정책 | Ant에서는 **실패 사례**를 보고함: 기술을 바꾸는 순간 넘어져서, 예시 5개 롤아웃이 기술을 6번 바꾸기 전에 모두 끝남(Fig 15) | 낮음 (참고: 넘어짐 문제) |
| 13 | Meta Learning Shared Hierarchies (MLSH) | Frans 외, 2018 | ICLR 2018 (동료 평가) | Gym Ant | 공유 하위 정책 + 메타러닝 | Ant Obstacle(보상이 희소한 장애물 코스): **MLSH Transfer 193 vs 단일 정책 0**. 넘어지지 않도록 관절을 주기적으로 리셋함 | 낮음 |
| 14 | Automatic Goal Generation (Goal GAN) | Florensa, Held 외, 2018 | ICML 2018, PMLR 80 (동료 평가) | Gym Ant (Free/Maze) | 자동 커리큘럼(적당히 어려운 목표를 생성) | Free Ant와 Maze Ant에서 기준선보다 빠르게 학습. 생성되는 목표가 점점 어려운 곳으로 옮겨감. 그래프만 있음 | 중간 (커리큘럼 원리) |
| 15 | Reverse Curriculum Generation | Florensa 외, 2017 | CoRL 2017 (동료 평가) | Gym Ant Maze | 목표 근처에서 시작해 시작 위치를 점점 멀리하는 커리큘럼 | 그냥 TRPO는 Ant Maze에서 학습이 "상당히 느려짐". 그래프만 있음 | 낮음 |
| 16 | Diversity is All You Need (DIAYN) | Eysenbach 외, 2019 | ICLR 2019 (동료 평가) | Gym Ant-v1 | 보상 없이 기술 찾기(탐색) | Ant는 점프와 여러 곡선 보행 기술을 배움("직선으로 걷는 기술은 없음"). 기술을 조합해 희소 보상 경유점 5개 과제를 품. 그래프만 있음 | 낮음 |
| 17 | Dynamics-Aware Unsupervised Discovery of Skills (DADS) | Sharma 외, 2020 | ICLR 2020 (동료 평가) | Gym Ant | 결과를 예측하기 쉬운 기술 찾기 + MPPI 계획 | Ant 기술이 "거의 뒤집히지 않고" 안정적. 학습 범위 밖의 목표에서 goal-conditioned RL보다 성능이 훨씬 덜 떨어짐. **다른 지형으로 옮기는 실험은 없음**. 그래프만 있음 | 낮음 |
| 18 | Model-Based Active Exploration (MAX) | Shyam 외, 2019 | ICML 2019, PMLR 97 (동료 평가) | Gym Ant Maze | 모델 앙상블로 능동 탐색 | U자 미로 끝까지 **40 에피소드(12k 스텝)** 만에 도달. 같은 시간에 기존 방법들은 중간쯤에 있음 | 낮음 |
| 19 | Maximum Diffusion RL | Berrueta 외, 2024 | Nature Machine Intelligence (동료 평가) | Gym/MuJoCo Ant | 경험끼리 상관을 줄이는 탐색, 한 번의 시도로 학습 | 뒤집히면 회복할 수 없다는 점 때문에 이론 조건이 깨지지만, Ant 단일 시도 학습에서도 SAC, NN-MPPI보다 좋음. 분산은 커짐. 그래프만 있음 | 낮음 |

---

## 2. 방법론별 정리

### A. 지형 다양화 / 지형 위 보행 (가장 관련 높음)

- **검증된 것**: Ant 비슷한 몸체(구동 관절 8개)를 **절차적으로 생성한 지형**에서 학습한 연구는 Heess 2017(#1) 하나였습니다. 확인한 내용은 두 가지입니다.
  1. 허들 지형에서 학습한 정책이 평지에서 학습한 정책보다 **처음 보는** 마찰, 울퉁불퉁한 표면, 경사, 구동기 힘 변화에서 더 강건한 경향이 있었습니다.
  2. 저자들은 이 사족 로봇의 "다리가 지형 변화에 비해 짧다"고 적었습니다.

  이 논문은 **arXiv에만 있고 동료 평가를 받지 않았으며**, 수치 없이 정규화한 그래프만 있습니다. 논문의 커리큘럼 실험(Fig 6a)은 캡션에 몸체가 적혀 있지 않아서 Ant 비슷한 몸체로 한 것인지 확인하지 못했습니다.
- **[추론] 우리 과제에 옮기면**: "평지만 학습하면 약하고, 거친 지형을 섞어 학습하면 처음 보는 변화에 더 강해진다"는 방향은 우리 기준선 결과(평지 117.8, 박스 지형 약 5)와 잘 맞습니다. 다리가 짧다는 지적은 **지형 높이 범위를 Ant 몸 크기에 맞춰 골라야 한다**는 근거로 쓸 수 있습니다.

### B. 강건성 / 도메인 랜덤화(DR) / 적대적 학습

- **검증된 것**: M2TD3(#5)는 Ant의 몸통과 다리 질량을 0.1~3배로 바꾸는 조건에서 **DR 방식이 RARL보다 훨씬 낫다**는 수치를 보여줍니다(Ant1 기준 DR 3.51, RARL −1.24). 최악 조건만 보면 M2TD3가 더 좋습니다. ACDR(#7, arXiv만)은 **PPO + Gym Ant**에서 구동기 고장 DR에 커리큘럼을 더하는 방식을 썼고, "어려운 것부터" 순서가 더 좋았다고 보고합니다. CARL(#4)은 Brax Ant의 마찰, 중력, 질량 등을 바꿀 수 있는 도구입니다.
- **주의(확인 결과)**: 사용자 목록에 있던 **RARL(Pinto 2017)과 EPOpt(Rajeswaran 2017)은 Ant를 쓰지 않습니다.** RARL은 InvertedPendulum, HalfCheetah, Swimmer, Hopper, Walker2d를 썼고, EPOpt는 Hopper와 HalfCheetah를 썼습니다.
- **[추론] 우리 과제에 옮기면**: TA가 마찰 같은 지형 파라미터를 바꾼다고 했으므로 **마찰과 질량 DR은 바로 해볼 만한 근거가 있습니다.** 다만 이 결과들은 모두 **평지에서 물리 값만 바꾼 것**이라, 지형 모양이 바뀌는 경우까지 효과가 있다는 근거는 아닙니다. RARL 방식은 Ant에서 불안정하다고 보고되었으므로 우선순위가 낮습니다.

### C. 적응 / 메타 RL

- **검증된 것**: Nagabandi/Clavera 2019(#9)는 Ant 다리 고장을 다룹니다. **학습 때 고장 낸 적 없는 다리**로 테스트해도 온라인 적응으로 대응했습니다. MAML(#8)과 PEARL(#10)의 Ant 과제는 **보상이 바뀌는 과제**(방향, 속도, 목표 위치)이고 물리나 지형이 바뀌는 과제가 아닙니다.
- **확인 결과**: UP-OSI(Yu 2017)는 Ant를 쓰지 않습니다(hopper 등). MOLe(Nagabandi 2019b, ICLR)의 본문에서도 Ant를 찾지 못했습니다. "Clavera 2019"는 #9와 같은 논문입니다(공동 1저자).
- **[추론] 우리 과제에 옮기면**: 이 방법들은 테스트 중에 모델이나 정책을 업데이트하거나, 과제마다 컨텍스트를 모으는 절차가 필요합니다. 그래서 TA가 play 스크립트로 한 번 실행해 평가하는 방식과 맞지 않습니다. 대신 "관측 기록(history)으로 암묵적으로 적응하는" 아이디어만 빌리는 것이 현실적입니다(PPO + 과거 관측 쌓기, 또는 RNN).

### D. 고장 / 몸체 변화에 대한 적응

- **검증된 것**: Ant 다리 고장은 #9(메타 RL)와 #7(DR + 커리큘럼, arXiv만)에 있습니다.
- **Ant 아님**: Cully 2015(Nature)는 **다리 6개 로봇(hexapod)**과 로봇 팔입니다(PDF에서 확인).
- **[추론]**: 지형 과제와 직접 관련은 없지만, "하나의 정책으로 여러 조건을 버티게 하는 DR"이라는 점에서 #7이 우리 설정(PPO, 단일 정책)과 가장 비슷합니다.

### E. 계층형 RL / 커리큘럼 (Ant 미로 계열)

- **검증된 것**: HIRO(#11)는 Ant Maze에서 성공률 0.99, Push 0.92, Fall 0.66을 기록했습니다. MLSH(#13)는 장애물 과제에서 193 대 0이었습니다. SNN4HRL(#12)은 Ant가 기술을 바꿀 때 넘어지는 실패를 보고했습니다. Goal GAN(#14)과 Reverse Curriculum(#15)은 Ant 미로에서 자동 커리큘럼이 효과가 있음을 보였습니다.
- **[추론]**: 모두 **평지 미로 길찾기**이고 지형 위 보행이 아닙니다. 그래서 직접 옮기기는 어렵습니다. 다만 Goal GAN의 "지금 정책에 **적당히 어려운** 과제를 고르는 방식"(성공률이 0.1~0.9 사이인 과제)은 지형 난이도 커리큘럼의 설계 근거로 인용할 수 있습니다. 또한 TA 평가 때 우리 커리큘럼 코드가 지형 교체 때문에 깨질 위험이 있으므로(메모), **학습용 설정과 제출용 설정은 분리해야 합니다.**

### F. 탐색 / 기술 찾기

- **검증된 것**: DIAYN(#16), DADS(#17), MAX(#18), MaxDiff(#19)는 모두 Ant에서 탐색이나 기술 다양성을 보여주었습니다. **Ant를 다른 지형으로 옮기는 실험은 넷 다 없습니다.**
- **[추론]**: 우리 과제에 기여할 가능성은 낮습니다. 굳이 쓴다면, 예측하기 쉬운 기술이 "거의 뒤집히지 않는다"는 DADS의 관찰 정도만 참고할 수 있습니다.

### G. 대규모 병렬 시뮬레이터 (Isaac Gym / Brax)

- **검증된 것**: Isaac Gym Ant는 A100 1장에서 20초에 보상 3000에 도달했고, 환경 8192개가 가장 좋았습니다(#2). Brax Ant는 약 10초였고 MuJoCo PPO는 약 30분이었습니다(#3). **둘 다 평지 Ant**입니다.
- **[추론]**: 우리 RTX 5060(8GB)은 A100보다 느리고 메모리도 적으므로 이 수치를 그대로 기대하면 안 됩니다. "Ant 평지 학습은 금방 끝난다"는 근거일 뿐이고, 지형 일반화에 대한 근거는 아닙니다.

### H. 일반화 벤치마크

- **확인 결과**: Packer 2018("Assessing Generalization in Deep RL")은 **Ant를 쓰지 않습니다**(HalfCheetah, Hopper 등, PDF에서 확인). CARL(#4)이 Ant에 물리 context를 제공하지만, Ant 실험 결과는 본문에 없습니다. **Ant로 절차적 지형을 만드는 동료 평가 벤치마크는 찾지 못했습니다.**

---

## 3. Ant 아님 (참고용, 표에서 제외)

| 논문 | 실제로 쓴 로봇/환경 | 근거 |
|---|---|---|
| RARL (Pinto 2017, ICML) | InvertedPendulum, HalfCheetah, Swimmer, Hopper, Walker2d | [arXiv:1703.02702](https://arxiv.org/abs/1703.02702) 초록과 본문 |
| EPOpt (Rajeswaran 2017, ICLR) | Hopper, HalfCheetah | [arXiv:1610.01283](https://arxiv.org/abs/1610.01283) 본문 |
| UP-OSI (Yu 2017, RSS) | hopper 등 (Ant 없음) | [arXiv:1702.02453](https://arxiv.org/abs/1702.02453) 본문 |
| Packer 2018 | HalfCheetah, Hopper 등 | [arXiv:1810.12282](https://arxiv.org/abs/1810.12282) 본문 |
| Cully 2015 (Nature) | 다리 6개 로봇, 로봇 팔 | [arXiv:1407.3501](https://arxiv.org/abs/1407.3501) 본문 |
| dm_control suite Quadruped "escape" (Tunyasuvunakool 외 2020) | 다리마다 구동기 3개, 총 **12개**라서 Ant(8개)와 다름. 다만 **무작위 산악 지형(heightfield) 탈출 과제**라서 지형 관점에서는 참고할 만함 | [arXiv:2006.12983](https://arxiv.org/abs/2006.12983) (확인한 것은 arXiv 버전) |
| SMERL (Kumar 2020, NeurIPS), MOLe (Nagabandi 2019, ICLR) | 본문에서 Ant를 찾지 못함 | [2010.14484](https://arxiv.org/abs/2010.14484), [1812.07671](https://arxiv.org/abs/1812.07671) |

---

## 4. 공백 (찾아봤지만 없었던 것)

1. **Teacher-student(특권 정보 학습)를 Ant에 적용해 지형 일반화를 본 논문**: 찾지 못했습니다. 이 방법의 근거는 모두 ANYmal, A1 같은 실제 사족 로봇 연구입니다(Ant 아님).
2. **Gym, Isaac, Brax Ant를 heightfield 거친 지형에서 평가한 동료 평가 논문**: 찾지 못했습니다. 가장 가까운 것은 Heess 2017(arXiv만, 8구동 DeepMind Quadruped)과 dm_control quadruped escape(12구동, Ant 아님)입니다.
3. **Ant 지형 난이도 커리큘럼**(예: Isaac Lab의 terrain_levels 방식): 동료 평가 논문을 찾지 못했습니다. Ant 커리큘럼 논문은 목표 위치나 시작 위치에 대한 커리큘럼이고(#14, #15), 고장 커리큘럼은 #7(arXiv만)입니다.
4. **Isaac Lab `Isaac-Ant-v0` 자체의 지형 일반화 결과**: Isaac Gym 논문(#2)에는 평지 Ant만 있습니다.
5. **학습 때 본 적 없는 지형 "유형"으로 zero-shot 전이한 Ant 결과(수치)**: 찾지 못했습니다. Heess 2017의 결과도 정규화한 그래프뿐입니다.
6. **DADS와 DIAYN의 Ant 기술을 다른 지형으로 옮긴 실험**: 해당 논문에는 없습니다.

**[추론] 결론**: "Ant + 처음 보는 지형 + PPO + zero-shot" 조합을 직접 다룬 검증된 선행 연구는 거의 없습니다. 그래서 우리 실험 결과는 선행 결과를 재현하는 것이 아니라 **새로운 측정**에 가깝고, 이 점을 독창성 항목에서 강조할 수 있습니다. 근거로 쓸 만한 것은 세 가지입니다.

- Heess 2017: 다양한 지형으로 학습하면 강건해짐 (동료 평가 아님이라고 밝히고 인용)
- M2TD3: Ant에서 DR이 RARL보다 나음
- ACDR: PPO + Ant에서 DR과 커리큘럼 사용 (동료 평가 아님)

---

## 5. 출처 URL

1. Heess 2017: https://arxiv.org/abs/1707.02286
2. Isaac Gym: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html , https://arxiv.org/abs/2108.10470
3. Brax: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/d1f491a404d6854880943e5c3cd9ca25-Paper-round1.pdf , https://arxiv.org/abs/2106.13281
4. CARL (TMLR): https://arxiv.org/abs/2202.04500 (PDF 머리말 "Published in TMLR 06/2023")
5. M2TD3: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2e0f5561c1553a97cee5fa64575358c9-Abstract-Conference.html , https://arxiv.org/abs/2211.03413
6. Tessler 2019: https://arxiv.org/abs/1901.09184 (PDF 머리말 "ICML 2019, PMLR 97")
7. ACDR: https://arxiv.org/abs/2111.10005
8. MAML: https://arxiv.org/abs/1703.03400 (PMLR 70)
9. Nagabandi/Clavera 2019: https://arxiv.org/abs/1803.11347 (PDF 머리말 "ICLR 2019")
10. PEARL: https://proceedings.mlr.press/v97/rakelly19a.html
11. HIRO: https://arxiv.org/abs/1805.08296 (NIPS 2018)
12. SNN4HRL: https://arxiv.org/abs/1704.03012 (PDF 머리말 "ICLR 2017"), Ant 실패 보고서: http://bit.ly/snn4hrl-antReport
13. MLSH: https://openreview.net/forum?id=SyX0IeWAW , https://arxiv.org/abs/1710.09767
14. Goal GAN: https://arxiv.org/abs/1705.06366 (PMLR 80)
15. Reverse Curriculum: https://arxiv.org/abs/1707.05300 (CoRL 2017)
16. DIAYN: https://openreview.net/forum?id=SJx63jRqFm , https://arxiv.org/abs/1802.06070
17. DADS: https://arxiv.org/abs/1907.01657 (PDF 머리말 "ICLR 2020"), https://sites.google.com/view/dads-skill
18. MAX: https://arxiv.org/abs/1810.12162 (PMLR 97), 코드: https://github.com/nnaisense/max
19. MaxDiff RL: https://www.nature.com/articles/s42256-024-00829-3 , https://arxiv.org/abs/2309.15293
