## initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter35000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### ami49(DeadSpace ratio = 3.7%)
#### initial condition(made by partial SA)
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





## initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter40000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### n100(DeadSpace ratio = 5.35%)
#### initial condition(made by partial SA)
![Figure_2](https://github.com/user-attachments/assets/58196af3-7466-4612-bd61-edab061e7cd7)
=== 부분 SA 후 Chip 상태 ===

경계 상자: W=431.00, H=523.00, 면적=225413.00

HPWL (절대값)             = 160022.00

정규화된 면적             = 627.888

정규화된 HPWL             = 188.850

정규화된 페널티          = 24.866

정규화된 DeadSpace       = 20.368

부분 SA 후 비용 (페널티만 사용) = 0.503


![Figure_4](https://github.com/user-attachments/assets/c495a763-be9d-4e46-8ad8-eefb5573c1f8)
![Figure_5](https://github.com/user-attachments/assets/6a7a48f2-7d32-4374-a523-128bd1aae3d8)

=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=417.00, H=492.00, 면적=205164.00

HPWL (절대값) = 143252.00, 정규화된 HPWL = 169.059

정규화된 면적 = 571.484, 정규화된 페널티 = 8.509

정규화된 DeadSpace = 12.509, 실제 DeadSpace = 25663.00 (12.51%)

비용 (모든 항 포함) = 1.444


![Figure_7](https://github.com/user-attachments/assets/bb0f85eb-4ac1-4e8e-9299-586ced4f5214)
![Figure_8](https://github.com/user-attachments/assets/0541baec-52b4-41a5-87cf-671b78591c6d)

=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=415.00, H=457.00, 면적=189655.00

최종 Compaction 후 HPWL (절대값)         = 146991.00

최종 Compaction 후 정규화된 면적         = 528.284

최종 Compaction 후 정규화된 HPWL         = 173.471

최종 Compaction 후 정규화된 페널티      = 0.000

최종 Compaction 후 정규화된 DeadSpace  = 5.354

최종 Compaction 후 실제 DeadSpace 면적 = 10154.00 (5.35%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 0.836




