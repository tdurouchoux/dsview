# dsview

Data science news monitoring management tools using LLM and graph representation in Obsidian

#TODO add try to session rollback ?

#TODO Remove commits while extraction is not fully successful
#TODO simplify extraction result management using relationships
#TODO Update write_notes to use less info > IN PROGRESS

#TODO Fix query_labels


#TODO Fix labels database (all labels are false now, almost the same results with mistral-small than with gpt4omini) > remake ER labels then opt

#TODO have some kind of quick documentation on the database (input content and labels should never be deleted, the rest is extraction results that can be recomputed )

> For structured output Mistral require the structure to be available in the prompt, what is the behavior with Instructor ?


#TODO distance between candidate topics and ER result

#TODO test langfuse on Onyxia
#TODO have some kind of auto evaluation to detect model drift > maybe using LangFuse

Next steps :

- Better interface :

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
