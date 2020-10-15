## Short description
*"Covid-19 AI demo in all-Docker"* deployment including dockerised Flask, FastAPI, Tensorflow Serving and HA Proxy etc etc.

## Scope
#### In scope:
As a jump start, we can simply use docker-compose to deploy the following dockerised components into an Ubuntu server

* *Univorn*  - web gateway server
* *FastAPI* - application server
* *Tensorflow-Serving* - application back-end servers for image etc classifications etc
* Python3 in *Jupyter Notebook* to emulate a client for benchmarking
* Docker and *docker-compose*

#### Out of scope or on next wish list:

* Ningx or Apache etc web servers are omitted in demo for now
* RabbitMQ and Redis  - queue broker for reliable messaging that can be replace by IRIS or Ensemble.   
* IAM (Intersystems API Manger) or Kong is on wish list
* SAM (Intersystems System Alert & Monitoring) 
* ICM (Intersystems Cloud Manager) with Kubernetes Operator - always one of my favorites since its birth
* FHIR (Intesystems IRIS based FHIR R4 server and FHIR Sandbox for SMART on FHIR apps)
* CI/CD devop tools or Github Actions

