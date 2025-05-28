## [hp] initial condition(made by partial SA) + 1st SA(iter1000) + 2st SA(iter3000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### hp(DeadSpace ratio = 4.07%)
#### initial condition(made by partial SA)
![Figure_2](https://github.com/user-attachments/assets/8a9cef7e-60f3-4e9d-a274-5205ca6eb9e1)
=== 부분 SA 후 Chip 상태 ===

경계 상자: W=6538.00, H=1554.00, 면적=10160052.00

HPWL (절대값)             = 0.00

정규화된 면적             = 575.276

정규화된 HPWL             = 0.000

정규화된 페널티          = 38.811

정규화된 DeadSpace       = 13.085

부분 SA 후 비용 (페널티만 사용) = 0.418

![Figure_4](https://github.com/user-attachments/assets/5d5a5fdf-5684-4d77-ade6-cc463059686b)
![Figure_5](https://github.com/user-attachments/assets/a8f1a4f1-07c4-4a16-8a25-eb80fe9eed3b)


=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=6160.00, H=1596.00, 면적=9831360.00

HPWL (절대값) = 0.00, 정규화된 HPWL = 0.000

정규화된 면적 = 556.665, 정규화된 페널티 = 31.151

정규화된 DeadSpace = 10.179, 실제 DeadSpace = 1000776.00 (10.18%)

비용 (모든 항 포함) = 1.213

![Figure_7](https://github.com/user-attachments/assets/c10255e7-41c4-4506-a6f9-ade40ef9b202)
![Figure_8](https://github.com/user-attachments/assets/53db3fa0-d02f-48a7-aef0-57a99284f68f)

=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=3304.00, H=2786.00, 면적=9204944.00

최종 Compaction 후 HPWL (절대값)         = 0.00

최종 Compaction 후 정규화된 면적         = 521.197

최종 Compaction 후 정규화된 HPWL         = 0.000

최종 Compaction 후 정규화된 페널티      = 0.178

최종 Compaction 후 정규화된 DeadSpace  = 4.067

최종 Compaction 후 실제 DeadSpace 면적 = 374360.00 (4.07%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 0.670

## [apte] initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter40000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### apte(DeadSpace ratio = 3.25%)
#### initial condition(made by partial SA)
![image](https://github.com/user-attachments/assets/cffbdbf2-d78a-4c45-af46-421de55cbe16)
=== 부분 SA 후 Chip 상태 ===

경계 상자: W=9438.00, H=5490.00, 면적=51814620.00

HPWL (절대값)             = 0.00

정규화된 면적             = 556.409

정규화된 HPWL             = 0.000

정규화된 페널티          = 2.929

정규화된 DeadSpace       = 10.138

부분 SA 후 비용 (페널티만 사용) = 0.370

![image](https://github.com/user-attachments/assets/1765ba27-d9f0-46b6-adcf-01bb9c3bced9)
![image](https://github.com/user-attachments/assets/0d382a31-500d-449c-86a7-940ae5d76c66)

=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=9438.00, H=5490.00, 면적=51814620.00

HPWL (절대값) = 0.00, 정규화된 HPWL = 0.000

정규화된 면적 = 556.409, 정규화된 페널티 = 2.929

정규화된 DeadSpace = 10.138, 실제 DeadSpace = 5252992.00 (10.14%)

비용 (모든 항 포함) = 1.181

![image](https://github.com/user-attachments/assets/576757b7-a043-4677-8b57-78dc11a35e2d)
![image](https://github.com/user-attachments/assets/2a28c3f2-72ae-4c8d-9769-597f3f4a87de)

=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=6578.00, H=7316.00, 면적=48124648.00

최종 Compaction 후 HPWL (절대값)         = 0.00

최종 Compaction 후 정규화된 면적         = 516.784

최종 Compaction 후 정규화된 HPWL         = 0.000

최종 Compaction 후 정규화된 페널티      = 0.000

최종 Compaction 후 정규화된 DeadSpace  = 3.248

최종 Compaction 후 실제 DeadSpace 면적 = 1563020.00 (3.25%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 0.601

## [ami33] initial condition(made by partial SA) + 1st SA(iter5000) + 2st SA(iter45000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### ami33(DeadSpace ratio = 2.7%)
#### initial condition(made by partial SA)
![1 (2 7%)](https://github.com/user-attachments/assets/01d883e4-1f5d-4df3-a8b3-96ec060f58d0)
=== 부분 SA 후 Chip 상태 ===

경계 상자: W=1323.00, H=1141.00, 면적=1509543.00

HPWL (절대값)             = 0.00

정규화된 면적             = 652.663

정규화된 HPWL             = 0.000

정규화된 페널티          = 5.804

정규화된 DeadSpace       = 23.391

부분 SA 후 비용 (페널티만 사용) = 0.437

![1-1 (2 7%)](https://github.com/user-attachments/assets/5c0fdd4c-6723-4874-8d57-89f3db1d32b2)
![1-2 (2 7%)](https://github.com/user-attachments/assets/f725f300-27df-4508-96fa-8b1dfdca375e)
=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=1148.00, H=1120.00, 면적=1285760.00

HPWL (절대값) = 0.00, 정규화된 HPWL = 0.000

정규화된 면적 = 555.909, 정규화된 페널티 = 0.000

정규화된 DeadSpace = 10.057, 실제 DeadSpace = 129311.00 (10.06%)

비용 (모든 항 포함) = 1.171

![1-3 (2 7%)](https://github.com/user-attachments/assets/8751129e-279d-4e0c-b887-2ed30ac15299)
![1-4 (2 7%)](https://github.com/user-attachments/assets/d1ce1c4a-5e25-4f9d-80b8-68ccad4ce5cd)
=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=1155.00, H=1029.00, 면적=1188495.00

최종 Compaction 후 HPWL (절대값)         = 0.00

최종 Compaction 후 정규화된 면적         = 513.855

최종 Compaction 후 정규화된 HPWL         = 0.000

최종 Compaction 후 정규화된 페널티      = 0.000

최종 Compaction 후 정규화된 DeadSpace  = 2.696

최종 Compaction 후 실제 DeadSpace 면적 = 32046.00 (2.70%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 0.555


## [ami49] initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter35000)
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





## [n100] initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter40000)
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






## [n200] initial condition(made by partial SA) + 1st SA(iter2000) + 2st SA(iter30000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### n200(DeadSpace ratio = 6.87%)
#### initial condition(made by partial SA)
![n200_initial(by_partial_SA)](https://github.com/user-attachments/assets/4c38569d-91bc-4704-ae2e-f76b08304b3f)

=== 부분 SA 후 Chip 상태 ===

경계 상자: W=502.00, H=501.00, 면적=251502.00

HPWL (절대값)             = 523290.00

정규화된 면적             = 715.731

정규화된 HPWL             = 624.211

정규화된 페널티          = 18.111

정규화된 DeadSpace       = 215.731

부분 SA 후 비용 (페널티만 사용) = 0.703


![n200_1st_SA(before_compact)](https://github.com/user-attachments/assets/cb2efdec-ba90-44e2-a549-03f8dbe7cb6e)
![n200_1st_SA(after_compact)](https://github.com/user-attachments/assets/44eb4549-fb3e-471f-8842-ec56fcc9c829)


=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=471.00, H=450.00, 면적=211950.00

HPWL (절대값) = 447508.00, 정규화된 HPWL = 533.814

정규화된 면적 = 603.173, 정규화된 페널티 = 1.591

정규화된 DeadSpace = 103.173, 실제 DeadSpace = 36254.00 (17.10%)

비용 (모든 항 포함) = 5.740


![n200_2nd_SA(before_compact)](https://github.com/user-attachments/assets/47a36edb-9087-47ab-8937-a36dc1c7ffa1)
![n200_2nd_SA(after_compact)](https://github.com/user-attachments/assets/de5e515f-9201-4ec5-a681-3e32cff23f97)


=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=474.00, H=398.00, 면적=188652.00

최종 Compaction 후 HPWL (절대값)         = 459802.00

최종 Compaction 후 정규화된 면적         = 536.871

최종 Compaction 후 정규화된 HPWL         = 548.479

최종 Compaction 후 정규화된 페널티      = 2.015

최종 Compaction 후 정규화된 DeadSpace  = 36.871

최종 Compaction 후 실제 DeadSpace 면적 = 12956.00 (6.87%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=50.00) = 2.386



## [n300] initial condition(made by partial SA) + 1st SA(iter5000) + 2st SA(iter45000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### n300(DeadSpace ratio = 7.42%)
#### initial condition(made by partial SA)
![21 n300(7 4%)](https://github.com/user-attachments/assets/e211b936-1c0d-4bb4-ac1d-f8c16ff05515)
=== 부분 SA 후 Chip 상태 ===

경계 상자: W=620.00, H=621.00, 면적=385020.00

HPWL (절대값)             = 803221.00

정규화된 면적             = 704.726

정규화된 HPWL             = 768.402

정규화된 페널티          = 93.677

정규화된 DeadSpace       = 29.050

부분 SA 후 비용 (페널티만 사용) = 0.820

![21-1 n300(7 4%)](https://github.com/user-attachments/assets/96866790-d9e2-48ca-aef8-27bbc0ef5927)
![21-2 n300(7 4%)](https://github.com/user-attachments/assets/68a08f1d-696f-4b78-bdd4-388807e4bf4c)

=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===

경계 상자: W=573.00, H=561.00, 면적=321453.00

HPWL (절대값) = 606866.00, 정규화된 HPWL = 580.559

정규화된 면적 = 588.375, 정규화된 페널티 = 0.283

정규화된 DeadSpace = 15.020, 실제 DeadSpace = 48283.00 (15.02%)

비용 (모든 항 포함) = 1.788


![21-3 n300(7 4%)](https://github.com/user-attachments/assets/19f81df8-63bc-4728-90dd-c18642503ed5)
![21-4 n300(7 4%)](https://github.com/user-attachments/assets/6eaed79b-19b1-4bc2-a53c-b711887df11f)

=== 최종 Compaction 후 (2단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=562.00, H=525.00, 면적=295050.00

최종 Compaction 후 HPWL (절대값)         = 655311.00

최종 Compaction 후 정규화된 면적         = 540.048

최종 Compaction 후 정규화된 HPWL         = 626.904

최종 Compaction 후 정규화된 페널티      = 0.000

최종 Compaction 후 정규화된 DeadSpace  = 7.416

최종 Compaction 후 실제 DeadSpace 면적 = 21880.00 (7.42%)

최종 Compaction 후 비용 (w_area=0.66, r_penalty=1.00, r_ds=80.00) = 1.163


