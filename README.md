# dsview

Data science news monitoring management tools using LLM and graph representation in Obsidian

Next steps :

- [x] Refactor ER to openai API
- [x] ER content extraction evaluation helper model
- [x] Content extraction evaluation
- [x] ER migration to new backend
- [x] Full migration to model_config and removal of langchain
- [x] Add new model providers
- [x] Migrate content_links
- [x] Model specific configuration
- [ ] Update configuration
- [ ] Streamlit vault db interface + Remove index logic
- [x] Add safeguard for duplicate content in inputcontent
- [ ] Fix update relevance, remove query in every stored url
- [ ] Add prediction table with title (and later cost ?)
- [ ] Git lfs dsview_vault migration
- [ ] Remove query from url > done only for new ingestion, original url will be kept in case of error in FailedIngestion table
- [ ] Better error notifications
- [ ] Faster queries - asynchronous parallel requests - add tests
- [ ] Query optimization with Dspy before implementation (if successfull)
- [ ] Fix issue with read medium
- [ ] Implement thinking query for LLMMode
- [ ] Get more info on try / retry
- [ ] Async ER run

Larger picture :

- Faster queries - asynchronous parallel requests
- Interact more with s3 for input and outputs (logs and config)
- Improve logging, add observability measures prices and consumption (ecologits), maybe try vector.dev (see calmcode for more info) (tokens + price + kWh) on ES openTelemetry ?
- Implement a newsletter-like mailing server + mail whenever the ingestion failed
- Improve PDF ingestion and remove token limit from content loaders
- Improve ER search for close topics, use semantic score + jaro
- Enable vector search on db > to explore
- Recommendation using relevance prediction

Next steps :

1. Quickly finish opt
2. Test again
3. Clean repo
4. Rerun full ingestion
5. Deploy

6. More labels
7. More opt + switch to mistral API (or claude for complex topics)

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

RAG on topics and content

TODO :
fix aysncio run when in api mode

v2 ? :

- s3
- Qdrant
