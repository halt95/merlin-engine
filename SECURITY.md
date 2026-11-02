<!-- Modified by the Merlin project from the upstream vLLM security policy; see NOTICE. -->
# Security policy

This policy covers the Merlin engine fork, its model recipes and its release tooling. It is maintained by the Merlin project and does not represent the upstream vLLM security team's policy.

## Report a vulnerability privately

Use [Merlin's private vulnerability report form](https://github.com/halt95/merlin-engine/security/advisories/new). Keep exploit details and sensitive data out of public issues and pull requests.

Include the affected Merlin version or commit, recipe and checkpoint revision, GPU/driver and dependency versions, the deployment configuration, the impact and a minimal reproducer. Remove credentials, private prompts, checkpoint access tokens and personal information from logs. Reports that also affect unmodified vLLM can be coordinated with [upstream vLLM](https://github.com/vllm-project/vllm/security/advisories/new).

The maintainer will assess reports and coordinate a fix and disclosure where appropriate. There is no guaranteed response deadline, bounty, CVE assignment or advance-notification programme.

## Versions

Merlin 3.0.1 is the current release. Security fixes target the latest public Merlin release; older v2.x recipe packages have no separate maintenance guarantee. Upgrade to the latest release and review its changelog. A release-candidate upstream base does not imply that every later upstream advisory has been fixed in Merlin.

## Deployment assumptions

Merlin inherits vLLM's trust assumptions. Read the bundled [upstream security guide](https://docs.vllm.ai/en/v0.32.0/usage/security.html), especially its guidance on distributed communication and media downloads.

- Keep distributed, worker and PLE communication on trusted, isolated networks. Do not expose those ports to the internet.
- Restrict access to the serving API and put authentication, TLS and resource controls at the deployment boundary.
- Treat model code, tokenizer/template code, build artifacts and serialized files as trusted inputs only after checking their provenance. Verify published checksums and review any use of remote model code.
- Restrict media URL destinations, local file access, download sizes and decode limits. A model's 8 MP encoder cap does not replace limits on downloading or decoding untrusted input.
- Avoid collecting private prompts or access tokens in shared logs. Apply dependency and driver security updates with appropriate requalification.

Hardware qualification and model-quality measurements are separate from security review. Neither the reference-host measurements nor a checksum guarantee that a deployment is secure.
