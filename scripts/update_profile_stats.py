#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from pathlib import Path


SCHOLAR_URL = "https://scholar.google.com/citations?user=ye4BnicAAAAJ&hl=en"
GITHUB_API_URL = "https://api.github.com/users/greatji"
GITHUB_REPOS_API_URL = "https://api.github.com/users/greatji/repos?per_page=100&type=owner&sort=updated"
DATA_PATH = Path("_data/profile.json")


def fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "greatji-homepage-stats-updater",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_citations(html: str) -> int:
    meta_match = re.search(r"Cited by ([0-9,]+)", html)
    if meta_match:
        return int(meta_match.group(1).replace(",", ""))
    raise ValueError("Could not find citation count in Google Scholar response")


def load_existing() -> dict:
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {}


def update_google_scholar(profile: dict) -> None:
    html = fetch_html(SCHOLAR_URL)
    citations = parse_citations(html)

    profile.setdefault("google_scholar", {})
    profile["google_scholar"]["url"] = SCHOLAR_URL
    profile["google_scholar"]["citations"] = citations
    profile["google_scholar"]["citations_display"] = f"{citations:,}"
    profile["google_scholar"]["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    profile["google_scholar"]["source"] = "google_scholar_profile"


def update_github(profile: dict) -> None:
    github_data = fetch_json(GITHUB_API_URL)
    repo_data = fetch_json(GITHUB_REPOS_API_URL)
    followers = int(github_data.get("followers", 0))
    public_repos = int(github_data.get("public_repos", 0))
    owned_non_fork_repos = [repo for repo in repo_data if not repo.get("fork")]
    total_stars = sum(int(repo.get("stargazers_count", 0)) for repo in owned_non_fork_repos)

    profile.setdefault("github", {})
    profile["github"]["url"] = github_data.get("html_url", "https://github.com/greatji")
    profile["github"]["followers"] = followers
    profile["github"]["followers_display"] = f"{followers:,}"
    profile["github"]["public_repos"] = public_repos
    profile["github"]["public_repos_display"] = f"{public_repos:,}"
    profile["github"]["owned_non_fork_repos"] = len(owned_non_fork_repos)
    profile["github"]["owned_non_fork_repos_display"] = f"{len(owned_non_fork_repos):,}"
    profile["github"]["total_stars"] = total_stars
    profile["github"]["total_stars_display"] = f"{total_stars:,}"
    profile["github"]["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    profile["github"]["source"] = "github_users_api+repos_api"


def main() -> int:
    profile = load_existing()

    try:
        update_google_scholar(profile)
        update_github(profile)
    except (HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to update profile stats: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected failure while updating profile stats: {exc}", file=sys.stderr)
        return 1

    DATA_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "Updated profile stats: "
        f"citations={profile['google_scholar']['citations']}, "
        f"github_followers={profile['github']['followers']}, "
        f"github_public_repos={profile['github']['public_repos']}, "
        f"github_total_stars={profile['github']['total_stars']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
