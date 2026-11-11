# g0v0-workflow-bot

## Review provider configuration

- `provider: str | None = "deepseek"` selects a built-in Pydantic AI provider and its model implementation.
- `review_base_url: str | None = None` overrides `provider` when supplied, using an OpenAI-compatible endpoint.
- `review_api_key` is passed to the selected provider. When omitted for a built-in provider, its normal environment-based authentication is used.
- If both `provider` and `review_base_url` are `None`, startup fails before review queues are restored.
- `review_request_limit` defaults to 1000 requests per iteration. The cumulative request limit is `review_request_limit × review_iteration` (1000 for iteration 1, 2000 for iteration 2, etc.), and historical request usage counts toward that cumulative limit. A positive integer changes the per-iteration allowance; `None` disables the limit.

For the built-in DeepSeek provider, leave `review_base_url` unset:

```dotenv
provider=deepseek
review_model=deepseek-flash
```

To use a custom gateway, set `review_base_url`; it takes priority over `provider`:

```dotenv
review_base_url=https://gateway.example.com/v1
review_api_key=your_api_key_here
review_model=your_model_here
```
