# dsview

Data science news monitoring management tools using LLM and graph representation in Obsidian

#TODO Remove commits while extraction is not fully successful
#TODO simplify extraction result management using relationships
#TODO Update write_notes to use less info > IN PROGRESS


Plan :
- reactivate push on update  > how does it work right now (need for way more git updates)
  - move it to streamlit interface
- rebuild vault
- check vault health
- update deployment for marimo
- deploy

#TODO Fix query_labels

#TODO compute missing distances or change representation for er comparison

#TODO Fix titles extraction

#TODO Fix labels database (all labels are false now, almost the same results with mistral-small than with gpt4omini) > remake ER labels then opt

#TODO have some kind of quick documentation on the database (input content and labels should never be deleted, the rest is extraction results that can be recomputed )

> For structured output Mistral require the structure to be available in the prompt, what is the behavior with Instructor ?


#TODO distance between candidate topics and ER result

#TODO test langfuse on Onyxia
#TODO have some kind of auto evaluation to detect model drift > maybe using LangFuse



Added tags and topic/content relation > implement ingestion + refacto obsidian

Add tests to evaluation

topics are defined :

- at the extraction result level ( + embedding)
- at the model level
- at the labels level ( - description)

> I could test out feeding a SQLModel, But I would still need the id (unless I decide that the name is a primary key). Works but still how would it work ??

Full restructuration (Cleaner code will kill me, but it might be worth it just for xp sake):

- models > models_utils
- llm_models > extraction/models + other place
- Central directory for all database schemas
- Obsidian creation launched via id calls (build note with id, build topic with id, ...) > avoid the need to forward extraction result and more agnostic to workflow. DB would be truly at the center

Use tach to trace this, just testing it

Search :

Commencer avec la création d'une interface de recherche
Utilisation de Qdrant + embedding de Concept ? ou Content ?

> Utilisation de duckdb in memory:

- sqlite fts5 only exact match of tokens
- sqlite-vec not really active dev, documentation still in beta
- sqlite would require finding ways to synchronize tables and virtual tables (a concern mostly for sqlite-vec)
- duckdb is fast to setup, can be build at run time only when needed
- duckdb is simpler to use, and has more search features (especially for fts)
- qdrant would have similar features but need to be managed + synchronized
- alternative would be pgvector + bm25 extension, could be done probably not worth it
- Most important issue storage of embedings as text in sqlite

> potential bottleneck from storing then querying array from sqlite

Search a topic :

- Find relevant topics and display graph

Ask some kind of question :

- Fetch relevant sources ?

Next steps :

- Better interface :

  - [ ] RAG like interface to fetch interesting content on a subject
  - [ ] Qwartz or equivalent for publishing + hosting ? Tried not full convincing, medium to poor presentation. still could be usefull as a backup (especially if storage is moved to s3). Need to think about what is usefull remotely (which features)
  - [ ] Newsletter like notifications (with relevance prediction ?)
  - [ ] Have some king of "watch" feature or remind me
  - [ ] Switch to marimo for dashboard interface
  - [ ] Migrate to Marimo ?

- Better data :

  - [ ] Follow RSS Feed of interesting publications (for example : Google research)- https://github.com/kurtmckee/feedparser or maybe scrap in some cases (Anthropic)

- Better data management :

  - [ ] Fix issue with missing note for some topics (could be related to that they already exist but under another type)
  - [ ] Migrate from git to S3 save state
  - [ ] Regular database dump to Sa3

- Better models :

  - [ ] Improve ER with vector db like search + jaro
  - [ ] Async ER run
  - [ ] Improve pdf ingestion and remove token limit from content loaders + better summarization
  - [ ] New LLM call architecture
  - [ ] Opt when sufficient number of annotations
  - [ ] Fix issue with read medium
  - [ ] Relevance prediction
  - [ ] Switch to Mistral
  - [ ] Better mlflow management (especially for  ing current parameters)

- Better management :
  - [ ] Better error notifications
  - [ ] Improve logging (with observability price consumption if possible) ES openTelemetry, vector.dev, ecologits

Model Maj 0.2:

- Entity Resolution : improvment made > added examples and more structured prompt
- content description : Done way better precision on tags
  - looking for good ratio title_rouge_f1 + tag_precision
  - contender : Anthropic 2.2e (good precision), haiku prompt v2 (anthropic, closer to default), Anthropic system prompt v1 (better f1)
  - Anthropic 2.2e final adjustements and testing reliability
- Topics extraction : Done, improvments
  - looking for system prompt
  - Found one, more recall a bit better in precision but more topics > check if it is not too much
- links extraction : no improvments

Generalization :

- more content types (represented differently)
- maybe need for support of tree structures (directory)
- privacy setting ? testing with small to really small langages models
- move from topic to Subjects > inferred by user with quickstart
- disappearance of tree structure ? add some kind of tag
  - would be a full datamanagement system
  - alternative to storing
