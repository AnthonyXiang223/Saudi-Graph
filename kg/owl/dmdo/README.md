# Disaster Management Domain Ontology (DMDO)
The Disaster Management Domain Ontology (DMDO) provides a semantic framework to align and connect various datasets on hazards and disasters contained within the [KnowWhereGraph](https://knowwheregraph.org/), a densely linked knowledge graph that integrates over 30 environmental and social datasets. 

DMDO offers: 
- a consistent knowledge pattern that can be used to query across all hazard datasets (e.g., on droughts, hurricanes and wildfires) in a uniform manner; 
- alignment of named events (e.g., *Hurricane Katrina*) across different datasets (e.g., NOAA Storm Events, FEMA Disaster Declarations Summaries, NOAA Historical Hurricane Tracks);
- methods to use or integrate data with authoritative classification schemas and vocabularies; and
- representation of causal relations between events and/or impacts and effects on people, places and public health.

DMDO consists of two modules: the [Hazard Event Module](./hazard-event-module/), which conceptualizes and organizes observational data pertaining to different types of hazards, largely by  re-using the [SOSA Ontology](https://www.w3.org/2015/spatial/wiki/SOSA_Ontology); and the [Hazard Operational Module](./hazard-operational-module/), which provides a pattern for operational effectiveness before, during and after an emergency.  

## Hazard Event Module

<img src="modules/hazard-event-module/hazard-event-module-overview.PNG" width=800 alt="Key concepts and properties in the phenomenon module. Orange boxes indicate concepts introduced in DMDO. Bright blue boxes indicate concepts from external ontologies (SOSA and GeoSPARQL here).">

[Click here for description of key concepts](#phenomenon-module)

## Operational Module
The Operational Module focuses on the **processual** and **functional** aspects of disaster management. 

**Processual aspects** are more activity oriented, taking one or more kinds of inputs (people or resources) to create outputs (or outcomes) that are of value in a given situation. Examples include mobilizing relief supplies, evacuating people, helping humanitarian responders to rapidly access to disaster victims for humanitarian responders, facilitate the timely flow of relief goods.

<img src="modules/operational-module/operational1.PNG" width=550 alt="High-level overview: Processual aspects">

**Functional aspects** may include the design, operation or improvement of systems and processes, for example, strengthening levee systems to prevent or reduce the severity of disasters. Functional aspects may also include the coordination, regulation and monitoring of relief and recovery assistance.

<img src="modules/operational-module/operational2.PNG" width=400 alt="High-level overview: Functional aspects">



## Contributors
TO ADD
## Publications
1. Stephen, S., Shimizu, C., Schildhauer, M., Zhu, R., Janowicz, K., and Hitzler, Pascal. "A Pattern for Representing Scientific Taxonomies." WOP@ ISWC (2022).
