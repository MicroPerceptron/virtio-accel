"""Decide whether a completed release workflow warrants a production refresh."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def eligible_run(event):
    run = event['workflow_run']
    return (run['conclusion'] == 'success'
            and run['head_repository']['full_name'] == event['repository']['full_name']
            and run['event'] in ('release', 'workflow_dispatch')
            and run['name'] in ('Publish crates', 'Release Python SDK'))


def published_to_pypi(jobs):
    # A successful artifact-only run (or TestPyPI upload) is not a release.
    return any(job['name'] == 'publish' and job['conclusion'] == 'success'
               and any(step['name'] == 'Publish to PyPI' and step['conclusion'] == 'success'
                       for step in job.get('steps', [])) for job in jobs)


def published_release(name, jobs):
    if name == 'Release Python SDK':
        return published_to_pypi(jobs)
    return any(job['name'] == 'publish-crates-io' and job['conclusion'] == 'success'
               for job in jobs)


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    refresh = True
    reason = 'Rebuild current main after a source change or manual request.'
    if os.environ['GITHUB_EVENT_NAME'] == 'workflow_run':
        refresh = eligible_run(event)
        reason = 'Release workflow failed or did not originate from this repository.'
        run = event['workflow_run']
        if refresh:
            jobs = []
            page = 1
            # Query this attempt, not an earlier success from a re-run. Never
            # download or execute artifacts or code from the triggering run.
            base = (f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}"
                    f"/actions/runs/{int(run['id'])}/attempts/{int(run['run_attempt'])}/jobs")
            while True:
                request = Request(f'{base}?per_page=100&page={page}', headers={
                    'Authorization': f"Bearer {os.environ['GH_TOKEN']}",
                    'Accept': 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                })
                with urlopen(request, timeout=30) as response:
                    batch = json.load(response)['jobs']
                jobs.extend(batch)
                if len(batch) < 100:
                    break
                page += 1
            refresh = published_release(run['name'], jobs)
            reason = ('Publication succeeded; rebuild current main.' if refresh else
                      'No production upload: artifact-only, TestPyPI, or skipped publication.')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'refresh={str(refresh).lower()}\n')
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write(f'## Website refresh\n\n{reason}\n')
    print(reason)


if __name__ == '__main__':
    main()
