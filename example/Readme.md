##initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter35000)
###partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty, where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

###ami49
####initial condition(made by partial SA)
![4ami49(2 7%)](https://github.com/user-attachments/assets/40ef2e4a-8de8-4caf-965f-dc62f2b07b37)
=== 부분 SA 후 Chip 상태 ===
경계 상자: W=6888.00, H=6258.00, 면적=43105104.00
HPWL (절대값)             = 0.00
정규화된 면적             = 608.049
정규화된 HPWL             = 0.000
정규화된 페널티          = 3.508
정규화된 DeadSpace       = 17.770
부분 SA 후 비용 (페널티만 사용) = 0.405


![4-1ami49(2 7%)](https://github.com/user-attachments/assets/e2029c06-63b2-4a8d-9c37-ef3e302b0fae)
![4-2ami49(2 7%)](https://github.com/user-attachments/assets/ac8e0234-09af-40f0-a753-66dea4afe72b)
=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===
경계 상자: W=6398.00, H=6090.00, 면적=38963820.00
HPWL (절대값) = 0.00, 정규화된 HPWL = 0.000
정규화된 면적 = 549.631, 정규화된 페널티 = 0.000
정규화된 DeadSpace = 9.030, 실제 DeadSpace = 3518396.00 (9.03%)
비용 (모든 항 포함) = 1.085


![4-3ami49(2 7%)](https://github.com/user-attachments/assets/2e8a9ec1-d63c-4600-9c21-2ec9a3e52e1f)
![4-4ami49(2 7%)](https://github.com/user-attachments/assets/cf6e63ec-7ea5-4aeb-8833-bd70aa17b6ea)
=== 최종 Compaction 후 (2단계 SA 결과 기반) ===
최종 Compaction 후 경계 상자: W=6174.00, H=5964.00, 면적=36821736.00
최종 Compaction 후 HPWL (절대값)         = 0.00
최종 Compaction 후 정규화된 면적         = 519.415
최종 Compaction 후 정규화된 HPWL         = 0.000
최종 Compaction 후 정규화된 페널티      = 0.000
최종 Compaction 후 정규화된 DeadSpace  = 3.738
최종 Compaction 후 실제 DeadSpace 면적 = 1376312.00 (3.74%)
최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 0.642


