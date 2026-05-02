# Security policy

## Reporting a vulnerability

If you discover a security issue in Henge, please **do not open a public GitHub issue**. Instead, report it privately so a fix can ship before the issue is disclosed.

- Email: open a [GitHub Security Advisory](https://github.com/ChrisPiz/Henge/security/advisories/new) on this repository (preferred — encrypted, ties directly to the codebase).
- Include: a description of the issue, the affected version or commit, reproduction steps, and the impact you observed.

You can expect an acknowledgement within 5 business days. We will work with you on a fix and a coordinated disclosure date.

## Scope

In scope:

- The Henge MCP server (`henge/` package, `server.py` entrypoint)
- The `setup` install script
- The bundled prompts and report renderer
- Anything that handles `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `VOYAGE_API_KEY`

Out of scope:

- Vulnerabilities in upstream dependencies (Anthropic SDK, OpenAI SDK, Voyage SDK, MCP SDK, scikit-learn, etc.) — please report those upstream
- Vulnerabilities in the host MCP client (Claude Code, Claude Desktop, Cursor) — report to those vendors
- Issues that require physical access to a user's machine after the keys have been entered into `.env`

## Hardening already in place

- `.env` and `.env.local` are gitignored — keys never leave the local machine
- The Quickstart paste prompt instructs the LLM agent to confirm only the *length* of pasted keys, never echo the value
- The `setup` script does not log keys, does not pass them on the command line, and does not write them anywhere except the user's local `.env`
- The MCP server validates keys at startup and exits with a clear stderr message if any are missing — it never prints the key value

## Security-related dependencies

Dependency updates that fix known vulnerabilities will be merged on a best-effort basis. We recommend running `pip install -U -e .` periodically and watching GitHub's Dependabot alerts on this repository.
