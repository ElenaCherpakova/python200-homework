<!-- Part 1: Warmup — Cloud Concepts -->

<!-- Cloud Concepts Question 1 -->
The core economic model of cloud computing is pay-as-you-go, which means you pay only for the cloud resources you use, such as storage, processing power, and networking.
This differs from owning your own servers because buying physical servers requires a large upfront investment and ongoing costs for maintenance, repairs, electricity, cooling, and infrastructure. With cloud computing, the cloud provider owns and maintains the physical hardware, so you can rent the resources you need without having to manage the physical servers yourself.

<!-- Cloud Concepts Question 2 -->

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means increasing the power of an existing server by adding more CPU, RAM, or other resources. For example, a small e-commerce business might vertically scale its server during the Christmas or New Year holiday season when there is a temporary increase in traffic.
Horizontal scaling means adding more servers or instances to distribute the workload. For example, a large digital content platform serving customers around the world might add servers in different regions to handle more users and reduce latency.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch. 
- Horizontal scaling — The web app suddenly needs to support many more users, so adding more servers allows the workload to be distributed across multiple machines.


A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM. 
- Vertical scaling — The model training job needs more computing power, so upgrading to a machine with a faster GPU and more RAM provides more resources on the same machine.

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines. 
- Horizontal scaling — The pipeline needs to process many more files, and because the work can be split across machines, multiple machines can process the files in parallel and distribute the workload.

<!-- Cloud Concepts Question 3 -->
Gmail - SaaS -  fully functional web app delievered directly to end-users without requiring them to manage the underlying infrastructure.
Azure Virtual Machines — IaaS — Provides virtual computing infrastructure where we have control over the operating system, applications, firewalls, and patches.
AWS S3 (Simple Storage Service) - IaaS - provides scalable object storage while AWS manages the underlying physical storage infrastructure.
GitHub Codespaces SaaS - ready to use, cloud-hosted development tool that works righ in the browser without requiring us to set up infrastructure.
Snowflake — SaaS — A fully managed data platform that abstracts away much of the underlying storage and computing infrastructure.
Supabase — BaaS — Provides a managed PostgreSQL database along with backend services such as authentication, APIs, and storage.

IaaS - infrastructure as a service, gives user maximum control by renting raw computing powerm storage and networking over the internet. An example is AWS EC2.
PaaS (Platform as a Service) provides a managed platform where developers can build, deploy, and run applications without managing the underlying servers and operating systems. Examples: Heroku and Google App Engine.
SaaS (Software as a Service) provides a complete, ready-to-use software application directly to end users. An example is Gmail.

<!-- Cloud Concepts Question 4 -->
Databricks and Snowflake are managed data platforms that provide data storage, processing, analytics, and other data-related tools without requiring users to manage the underlying infrastructure.
The difference from using a cloud provider like AWS or GCP directly is the level of abstraction. With AWS or GCP, we can work with lower-level infrastructure and configure services such as virtual machines, storage, networking, and databases ourselves. With a managed data platform, many of these components are pre-configured and managed for us.
We gain simplicity, faster development, and less infrastructure management, but we give up some control and flexibility over the underlying infrastructure and may become more dependent on the platform.

<!-- Cloud Concepts Question 5 -->
The cloud is probably not the right choice when the dataset fits comfortably on a single machine and there are no massive compute demands, because local processing can be faster and cheaper. It may also not be the right choice when the learning curve and complexity of cloud infrastructure outweigh the benefits, especially for simple tasks or initial prototypes.

<!-- Part 2: Warmup — Cloud Landscape -->
<!-- Cloud Landscape Question 1 -->
There are three hyperscalers: Amazon Web Service (AWS) - is the oldest and largest. Has a broadest service catalog of any provider. Great fit if you're working in a large enterprise, a startup or a non-profit with engineering staff.
GCP (Google Cloud Plaform) - is strongest in data and machine learning. Built many of the foundation ideas in modern distributed systems. Great fit if you're working on large-scale analytics or ML infrastructure.
Microsoft Azure - is the dominant provider in enterprise and goverment settings, largely because of its deep integreation with Windows, Active Directory and Microsoft 365. Good fit if you are large non-profit and public sector organizations and if you are already Azure customer.

<!-- Cloud Landscape Question 2 -->
Access: Supabase is easier to access because students can create their own accounts in a few minutes without needing organizational provisioning or waiting for invitations.
Pedagogical fit: Supabase uses a relational database with rows and columns, which makes skills like querying and working with structured data more transferable to other data roles.
Pipeline coherence: Supabase makes it easy to represent the raw and enriched stages of the ETL pipeline as related tables, making the pipeline easier to inspect and debug.

Reflection: When starting a new project, I should evaluate a cloud tool based not only on its features, but also on accessibility, how well it fits the project's goals, how easy it is to manage, and whether it makes the development process simpler.

<!-- Cloud Landscape Question 3 -->
1. You need to store 10 TB of image files and retrieve them by filename from any machine. - Object storage (Amazon S3)
2. You need to run an ML training job on a GPU for four hours, then shut it down. - ML Platform (SageMaker)
3. You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets. - Serverless Compute (Amazon Lambda)
4. You need to send structured data to a large language model and get a text response back. - LLM API (Azure OpenAI)

<!-- Cloud Landscape Question 4 -->
I could build an e-commerce application where the application runs on AWS EC2, product images and other static content are stored in Amazon S3, and customer, product, and order data are managed using Azure SQL Database.
Consolidating everything to one provider could simplify management, billing, networking, and authentication because the services are designed to work together. However, I would give up the flexibility to choose the best service from different providers and could become more dependent on a single provider.