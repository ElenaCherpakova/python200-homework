I confirmed that project was set up with no issue. 

Scenario A, using a t3.micro EC2 instance for approximately 160 hours per month, costs $1.66 per month, or $19.92 for 12 months. I was surprised by how inexpensive the lightweight instance is. Scenario B costs approximately $2,570.16 per month, or $30,841.92 for 12 months, including the p3.2xlarge EC2 instance, RDS db.m5.large, and 1 TB of S3 Standard storage. While exploring the calculator, I found it interesting how much the cost changes depending on the instance type and the amount of resources, especially when a GPU is included. The large difference between the two scenarios shows that a GPU instance can be worth the cost for workloads that genuinely require GPU computing, such as ML training, but it would be unnecessarily expensive for lightweight workloads that do not need a GPU.

Link: https://www.loom.com/share/b3d7e6381c7f4e6e8acb7cf37dfb99d3
