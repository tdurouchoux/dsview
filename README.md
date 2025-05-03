# dsview

Data science news monitoring management tools using LLM and graph representation in Obsidian

Next steps :

- Better interface :

  - [ ] RAG like interface to fetch interesting content on a subject
  - [ ] Qwartz or equivalent for publishing + hosting ?
  - [ ] Newsletter like notifications (with relevance prediction ?)
  - [ ] Have some king of "watch" feature or remind me
  - [ ] Switch to marimo for dashboard interface

- Better data management :

  - [ ] Fix issue with missing note for some topics (could be related to that they already exist but under another type)
  - [ ] Migrate from git to S3 save state

- Better models :

  - [ ] Improve ER with vector db like search + jaro
  - [ ] Async ER run
  - [ ] Improve pdf ingestion and remove token limit from content loaders + better summarization
  - [ ] New LLM call architecture
  - [ ] Opt when sufficient number of annotations
  - [ ] Fix issue with read medium
  - [ ] Relevance prediction
  - [ ] Switch to Mistral
  - [ ] Better mlflow management (especially for selecting current parameters)

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
