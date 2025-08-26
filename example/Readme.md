## [hp] initial condition(made by partial SA) + 1st SA(iter1000) + 2nd SA(iter3000)
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

## [apte] initial condition(made by partial SA) + 1st SA(iter2000) + 2nd SA(iter40000)
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

## [ami33] initial condition(made by partial SA) + 1st SA(iter5000) + 2nd SA(iter45000)
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


## [ami49] initial condition(made by partial SA) + 1st SA(iter2000) + 2nd SA(iter35000) 
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

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





## [n100] initial condition(made by partial SA) + 1st SA(iter3000) + 2nd SA(iter5000) + 3rd SA(40000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### n100(DeadSpace ratio = 3.42%)
#### initial condition(made by partial SA)



<img width="2983" height="1540" alt="floorplan_1st_K-Parent_SA_Result" src="https://github.com/user-attachments/assets/7d6157a2-af99-4c1f-8c24-0c23e0705213" />

=== 1단계 SA 후 상태 ===

[MP K-Parent Iter= 3000] T=  0.0000 | Cost_avg=   0.361 | Best=   0.341 



<img width="2977" height="1537" alt="floorplan_2nd_K-Parent_SA_Result" src="https://github.com/user-attachments/assets/1d1d42bd-0b77-42ec-8ed5-31b52f2fcab8" />

=== 2단계 SA 후 상태===

[MP K-Parent Iter= 5000] T=  0.5529 | Cost_avg=  17.970 | Best=   5.205 



=== 3단계(최종) SA + Compaction 후 ===
<img width="2977" height="1537" alt="floorplan_3rd_Single_SA_Result" src="https://github.com/user-attachments/assets/43dbfe38-0f03-4a76-bfc7-cb8bf1842a59" />
<img width="2977" height="1537" alt="floorplan_Final_Compacted_Layout" src="https://github.com/user-attachments/assets/885e8e12-a296-41b8-ae59-d155bf45d02c" />

=== 최종 Compaction 후 (3단계 SA 결과 기반) ===

최종 Compaction 후 경계 상자: W=450.00, H=413.00, 면적=185850.00

최종 Compaction 후 HPWL (절대값)         = 239662.00

최종 Compaction 후 스케일링된 면적 항    = 18.140

최종 Compaction 후 스케일링된 HPWL 항    = 42.195

최종 Compaction 후 스케일링된 페널티 항  = 0.000

최종 Compaction 후 스케일링된 DeadSpace 항= 4.142

최종 Compaction 후 실제 DeadSpace 면적 = 6349.00 (3.42%)

최종 Compaction 후 비용 (w=0.66, r_penalty=15.00, r_ds=96.00) = 4.239






## [n200] initial condition(made by partial SA) + 1st SA(iter3000) + 2nd SA(iter3000) + 3rd SA(iter 40000)
### partial SA and 1st SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

### n200(DeadSpace ratio = 4.88%)
#### initial condition(made by partial SA)
<img width="2889" height="1786" alt="floorplan_Initial_Random" src="https://github.com/user-attachments/assets/3701c67f-0ee6-43d5-a4d7-de69f27088fb" />


=== 부분 SA 후 Chip 상태 ===

경계 상자: W=377.00, H=530.00, 면적=199810.00

HPWL (절대값)             = 411416.00

스케일링된 면적 항        = 35.116

스케일링된 HPWL 항        = 33.796

스케일링된 페널티 항      = 0.031

스케일링된 DeadSpace 항   = 17.460

부분 K-Parent SA 후 비용 (페널티만 사용) = 0.347


<img width="2923" height="1505" alt="floorplan_1st_K-Parent_SA_Result" src="https://github.com/user-attachments/assets/bfa859cb-a2b5-46b3-a763-67cfb72104aa" />


=== 1단계 SA 후 상태 (참고용 Dead Space 포함 비용) ===

[MP K-Parent Iter= 3000] T=  0.0001 | Cost_avg=   0.473 | Best=   0.454 | Improved: True


<img width="2930" height="1509" alt="floorplan_2nd_K-Parent_SA_Result" src="https://github.com/user-attachments/assets/69fbf4dd-0e6c-4190-aef0-f7b696dcae49" />


=== 2단계 SA 후 상태 (참고용 Dead Space 포함 비용) ===


[MP K-Parent Iter= 3000] T=  0.6829 | Cost_avg=  24.083 | Best=   9.429 | Improved: False


<img width="2948" height="1519" alt="floorplan_Final_Compacted_Layout" src="https://github.com/user-attachments/assets/a70bd326-9485-41f6-b39f-4aec14c543e6" />


=== 3단계 단일 심층 SA (최종 미세 조정) + Compaction 후 상태 ===

최종 Compaction 후 경계 상자: W=384.00, H=481.00, 면적=184704.00

최종 Compaction 후 HPWL (절대값)         = 448512.00

최종 Compaction 후 스케일링된 면적 항    = 32.461

최종 Compaction 후 스케일링된 HPWL 항    = 36.843

최종 Compaction 후 스케일링된 페널티 항  = 0.003

최종 Compaction 후 스케일링된 DeadSpace 항= 7.056

최종 Compaction 후 실제 DeadSpace 면적 = 9008.00 (4.88%)

최종 Compaction 후 비용 (w=0.66, r_penalty=15.00, r_ds=96.00) = 7.113



## [n300] initial condition(made by partial SA) + 1st SA(iter5000) + 2nd SA(iter45000)
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



## [n300] initial condition(made by partial SA) + 1st SA(iter5000) + 2nd SA(iter5000) + 3rd SA(iter40000) (할당된 워커: 10개)
### partial SA and 1st, 2nd, 3rd SA's cost function use AREA, HPWL, penalty(exceed the standard chip penalty)

(where standard chip is square which has width, height = sqrt(all modules area sum*1.2) )

[R1]
<img width="2982" height="1539" alt="floorplan_Final_Compacted_R1 0" src="https://github.com/user-attachments/assets/d086904c-c852-48db-8530-3fb26bfa91c9" />

--- 결과 (목표 R = 1.0) ---
최종 경계 상자: W=551.00, H=535.00, 면적=294785.00 (실제 R: 0.97)
최종 HPWL (절대값)         = 606383.00
최종 실제 DeadSpace 면적 = 21615.00 (7.33%)
최종 비용 (w=0.66, r_penalty=2000.00, r_ds=100.00) = 10.138

[R2]
<img width="2701" height="1486" alt="floorplan_Final_Compacted_R2 0" src="https://github.com/user-attachments/assets/30f13b65-114f-4e2b-b658-3c8e2c96ceb9" />

--- 결과 (목표 R = 2.0) ---
최종 경계 상자: W=396.00, H=738.00, 면적=292248.00 (실제 R: 1.86)
최종 HPWL (절대값)         = 687826.00
최종 실제 DeadSpace 면적 = 19078.00 (6.53%)
최종 비용 (w=0.66, r_penalty=2000.00, r_ds=100.00) = 9.066

[R3]
<img width="2581" height="1486" alt="floorplan_Final_Compacted_R3 0" src="https://github.com/user-attachments/assets/92e05819-5272-4f27-9b19-5c130f109f4e" />

--- 결과 (목표 R = 3.0) ---
최종 경계 상자: W=335.00, H=890.00, 면적=298150.00 (실제 R: 2.66)
최종 HPWL (절대값)         = 738672.00
최종 실제 DeadSpace 면적 = 24980.00 (8.38%)
최종 비용 (w=0.66, r_penalty=2000.00, r_ds=100.00) = 11.575


