# Gregg Shorthand Lookup: Render security deployment

## What is affected

This is a Flask/Python service. PHP2Shell and Node runtime patches are not applicable unless another build step uses PHP or Node. The eMudhra trust-store issue applies to outgoing HTTPS requests and build-time downloads.

## Deployment checklist

1. Put `app.py` in the repository root, `index.html` in `templates/`, and `style.css` in `static/`. Keep all existing SVG assets.
2. Replace root `requirements.txt` with the provided file. This adds `certifi>=2026.10.19`.
3. **Important:** The public GitHub repository also contains `requirements.lock.txt` with older pins and no Certifi. If Render's build command installs from that lockfile, regenerate it from the new requirements or switch the build command to `pip install -r requirements.txt`; otherwise the update is not applied. Do not assume updating requirements.txt alone is sufficient. Check the resolved environment with `python -m pip show certifi`.
4. Deploy with a full **rebuild** (not a restart). The native Python build refreshes the system CA store. If this service uses Docker instead, update the base image and install updated `ca-certificates` in the Dockerfile.
5. Render start command: `gunicorn app:app --bind 0.0.0.0:$PORT` (retain your existing working command if equivalent). Do not use Flask's development server in production.
6. If the build uses Node, separately upgrade it to Render's patched versions (`22.23.4`, `24.22.1`, `26.12.1` or later in the same major). This Flask code itself does not need Node.
7. Test `/`, `/lookup?word=the`, `/suggest?word=th`, `/segment?word=the`, and your Major numerical lookup. Confirm Content-Security-Policy in response headers.
8. For a true certificate rejection test, use an HTTPS client **inside the running service**, as described in Render's advisory. The present application does not make outbound HTTPS requests in its normal lookup routes, apart from NLTK corpus retrieval when WordNet is absent.

## Notes and limitations

* Added Content Security Policy uses a SHA-256 hash for the existing inline JavaScript. **If you change that inline script**, recalculate the hash and update `INLINE_SCRIPT_SHA256` in `app.py`; otherwise the browser will block the script.
* The application still downloads NLTK WordNet data at runtime if it is missing. Prefer installing these datasets at build time in the deployment environment, so that normal requests do not need external network access.
* Pinned or lower-bounded requirements are not a complete security audit. Review transitive packages and regenerate any full dependency lock in CI.
* These changes have not been committed to GitHub or deployed to Render.

Render advisory: https://render.com/docs/advisory-emudhra
