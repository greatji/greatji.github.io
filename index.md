---
layout: default
title: Home
permalink: /
---
#### I received my Ph.D in database group of [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), under the supervision of professor [Guoliang Li](http://dbgroup.cs.tsinghua.edu.cn/ligl/).I received my bachelor degree in Computer Science of [BUPT](http://www.bupt.edu.cn/). I am interested in Distributed System, Similarity Search and Machine Learning for Query Optimization.   
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:18px;">
  <a href="{{ site.data.profile.google_scholar.url }}" style="display:inline-flex;align-items:center;gap:6px;">
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" style="color:#1f5fbf;">
      <path fill="currentColor" d="M12 3 1 9l11 6 9-4.91V17h2V9L12 3Zm-7 9.18V16l7 3.82L19 16v-3.82l-7 3.82-7-3.82Z"/>
    </svg>
    <span>Google Scholar (Cited by {{ site.data.profile.google_scholar.citations_display | default: site.data.profile.google_scholar.citations }})</span>
  </a>
  <a href="{{ site.data.profile.github.url }}" style="display:inline-flex;align-items:center;gap:6px;">
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" style="color:#24292f;">
      <path fill="currentColor" d="M12 .5C5.65.5.5 5.66.5 12.02c0 5.09 3.29 9.4 7.86 10.92.58.11.79-.25.79-.56 0-.28-.01-1.2-.02-2.18-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.78 2.71 1.26 3.37.97.1-.75.4-1.26.72-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.17 1.17a10.9 10.9 0 0 1 5.78 0c2.2-1.48 3.16-1.17 3.16-1.17.63 1.59.24 2.76.12 3.05.73.8 1.17 1.82 1.17 3.07 0 4.41-2.69 5.39-5.25 5.67.41.35.77 1.04.77 2.1 0 1.52-.01 2.74-.01 3.11 0 .31.21.68.8.56A11.52 11.52 0 0 0 23.5 12C23.5 5.66 18.35.5 12 .5Z"/>
    </svg>
    <span>GitHub ({{ site.data.profile.github.public_repos_display | default: "0" }} repos, {{ site.data.profile.github.total_stars_display | default: "0" }} stars)</span>
  </a>
</div>

## Work Experience 
- 2026.03 - present Assistant Professor at Tsinghua University, Haidian, Beijing  
- 2021.07 - 2025.07 Researcher at Huawei, Haidian, Beijing  
  Works on build-in AI for openGauss/GaussDB (VectorDB, AI-based Optimizer, LLM, NewHardware)  
- 2017.11 - 2018.05 Visit Researcher of CSAIL, [MIT](https://www.csail.mit.edu/), Boston, MA  
  Works on end-to-end entity consolidation framework  

## News!
- 2026.6.1 We open-source our vector cores on PostgreSQL and DuckDB for more convenience deployment [VexDB-Lite](https://github.com/VexDB-THU/VexDB-Lite)  
- 2025.9.15 We release our commercial vector database product [VexDB v3.0.0](https://www.vexdb.com)  
- 2025.8.7  Our data agent system AgenticData got the top score on both [DABstep Benchmark](https://huggingface.co/spaces/adyen/DABstep) and [Spider-2.0-Lite](https://spider2-sql.github.io/)

## Researches

- SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search.  
Yuchen Peng, Dingyu Yang, Zhongle Xie, **Ji Sun**, Lidan Shou, Ke Chen, Gang Chen  
[[Full Research(VLDB2026)](To appear)]

- A Topology-Aware Localized Update Strategy for Graph-Based ANN Index.  
Song Yu, Shengyuan Lin, Shufeng Gong, Yongqing Xie, Ruicheng Liu, Yijie Zhou, **Ji Sun**, Yanfeng Zhang, Guoliang Li, Ge Yu  
[[Full Research(VLDB2026)](To appear)]

- GaussDB-Vector: A Large-Scale Persistent Real-Time Vector Database for LLM Applications.  
**Ji Sun**, Guoliang Li, James Jie Pan, Jiang Wang, Yongqing Xie, Ruicheng Liu, Wen Nie  
[[Full Industry(VLDB2025)](resource/p2114-li (1).pdf)] [[Slides](resource/gaussdb-vector.pdf)]  

- Boosting Accuracy and Efficiency for Vector Retrieval with Local Scaling Graph.  
Hongya Wang, WenLong Wu, Cong Luo, Aobei Bian, Chunguang Meng, Yishuo Wu, **Ji Sun**  
[[Full Research(ICDE2025)](To appear)]

- GaussML: An End-to-End In-Database Machine Learning System.  
Guoliang Li, **Ji Sun**, Lijie Xu, Shifu Li, Jiang Wang, Wen Nie  
[[Full Industry(ICDE2024)](resource/gaussmlicde.pdf)]  [[Slides](resource/GaussML.pptx)]

- Learned Cardinality Estimation: A Design Space Exploration and a Comparative Evaluation.  
**Ji Sun**, Jintao Zhang, Zhaoyan Sun, Guoliang Li, Nan Tang  
[[Code](https://github.com/jt-zhang/CardinalityEstimationTestbed)] [[E&A(VLDB2022)](http://da.qcri.org/ntang/pubs/[vldb22]learned.cardinality.pdf)] [[Slides](resource/Research_321.pptx)]

- DBMind: A Self-Driving Platform in openGauss.  
Xuanhe Zhou,Lianyuan Jin,**Ji Sun**,Xinyang Zhao,et al.  
[[Demo(VLDB2021)](http://vldb.org/pvldb/vol14/p2743-zhou.pdf)]

- openGauss: An Autonomous Database System.  
Guoliang Li,Xuanhe Zhou,**Ji Sun**,Xiang Yu,et al.  
[[Full Industry(VLDB2021)](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/vldb21-opengauss.pdf)]

- Learned Cardinality Estimation for Similarity Queries.  
**Ji Sun**, Guoliang Li, Nan Tang  
[[Full Research(SIGMOD2021)](https://dl.acm.org/doi/pdf/10.1145/3448016.3452790)] [[Slides](resource/SIGMOD21-fp36.pdf)]   

- An Autonomous Materialized View Management System with Deep Reinforcement Learning.  
Yue Han, Guoliang Li, Haitao Yuan, **Ji Sun**  
[[Short (ICDE2021)](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/view-icde21.pdf)]   

- Query Performance Prediction for Concurrent Queries using Graph Embedding.  
Xuanhe Zhou, **Ji Sun**, Guoliang Li, Jianhua Feng  
[[Full Research(VLDB2020)](http://www.vldb.org/pvldb/vol13/p1416-zhou.pdf)]   

- Automatic View Generation with Deep Learning and Reinforcement Learning.  
Haitao Yuan, Guoliang Li, Ling Feng, **Ji Sun**, Yue Han  
[[Full Research(ICDE2020)](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b501/290300b501.pdf)]    

- An End-to-End Learning-based Cost Estimator.  
**Ji Sun**, Guoliang Li  
[[Code](https://github.com/greatji/Learning-based-cost-estimator)] [[Full Research(VLDB2020)](http://www.vldb.org/pvldb/vol13/p307-sun.pdf)][[slide](resource/vldb20.pdf)]   

- Database Meets Artificial Intelligence: A Survey.  
Xuanhe Zhou, Chengliang Chai, Guoliang Li, **Ji Sun**  
[[Full Research(TKDE2020)](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/aidb.pdf)]

- Hybrid Entity Resolution Approach for E-commerce Products.  
**Ji Sun**, Guoliang Li  
[[Code](https://github.com/greatji/SigmodProgrammingContest2020_code)] [[Poster(Sigmod2020)](http://www.inf.uniroma3.it/db/sigmod2020contest/posters/DBTHU_sigmod_programming_contest_2020_poster.pdf)]  

- Optimizing Human Involvement for Entity Matching and Entity Consolidation.  
**Ji Sun**, Dong Deng, Ihab Ilyas, Guoliang Li, Samuel Madden, Mourad Ouzzani, Michael Stonebraker, Nan Tang  
[[Technical Report](http://arxiv.org/abs/1906.06574)]  

- Dima: Distributed In-memory Similarity-based Query Processing System.  
**Ji Sun**, Zeyuan Shang, Guoliang Li, Dong Deng, Zhifeng Bao  
[[Code](https://github.com/TsinghuaDatabaseGroup/Dima.git)] [[Video](https://youtu.be/oJmNKK0O67U)] [[Demo(VLDB2017)](http://www.vldb.org/pvldb/vol10/p1925-sun.pdf)] [[Full Research(VLDB2019)](http://www.vldb.org/pvldb/vol12/p961-sun.pdf)]  

## Selected Awards
- Excellent Doctoral Dissertation Award, China Computer Federation(CCF), 2021
- Outstanding Graduate of Beijing, Beijing Gov., 2021
- Outstabding Graduate, Department of Computer Science of Tsinghua, 2021
- 84 Future Innovation Scholarship, Tsinghua University, 2020
- Xiaomi Scholarship, Tsinghua University, 2020
- ACM Sigmod Programming Contest [Finalist](resource/Ji_Sun.pdf), SIGMOD, 2020
- Student Travel Award, VLDB, 2019
- Mei Yiqi Scholarship, Tsinghua University, 2017, 2019
- Outstanding Graduate of Beijing, Beijing Gov., 2016
- First Prize Scholarship, BUPT, 2014 & 2015
- National Scholarship, Ministry of Education, 2013

## Talks
- openGauss：构建内外兼修的数据库智能自治能力 [ADL@CCF][[Slides]()]
- openGauss：构建自学习型智能内核 [DTC][[Slides]()]
- What Do We Really Need for Vector Database [CloudDB Workshop @ VLDB2024][[Slides](resource/vectordb-keynote-vldb24.pdf)]  
- Waking up dormant data: techniques of multimodal data analysis [CSC 696J@Arizona State University][[Slides]()]
- 数据智能体：自主数据处理的新范式 [DACon@DataFun][[Slides]()]

## Services
- PC Member of ICDE 2027  
- Committee Member of pVLDB Reproducibility  
- Review of ACM TODS  
- Review of JCST  

_____
<div style="height:100px;width:200px;margin:0 auto">
<center>
<script type="text/javascript" id="clustrmaps" src="//cdn.clustrmaps.com/map_v2.js?d=pe1rfPbhTfzky5ISQu4qQ1Xwqj7y_bFjS5d6afAShdk&cl=ffffff&w=a"></script>
</center>
</div>
