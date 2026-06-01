# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Jekyll site for an academic homepage. Root content lives in [index.md](/Users/sunji/Work/greatji.github.io/index.md) and site settings live in [_config.yml](/Users/sunji/Work/greatji.github.io/_config.yml). Blog posts are stored in `_posts/` using the Jekyll pattern `YYYY-MM-DD-title.markdown` or `.html`. Layout overrides belong in `_layouts/`. Static images go in `figures/`, and downloadable papers or slides go in `resource/`. Keep assets referenced with repo-relative paths such as `/figures/model.png` or `/resource/gaussdb-vector.pdf`.

## Build, Test, and Development Commands
Use standard Jekyll commands from the repository root.

- `bundle install` installs Ruby and Jekyll dependencies if a `Gemfile` is present or added locally.
- `bundle exec jekyll serve` runs the site locally with live reload, usually at `http://127.0.0.1:4000`.
- `bundle exec jekyll build` generates the static site in `_site/` for validation before publishing.

If Bundler is not configured in this clone, install Jekyll first and then rerun the same commands through `bundle exec`.

## Coding Style & Naming Conventions
Prefer Markdown for content pages and posts, with short paragraphs and consistent heading levels. Use YAML front matter on every post, for example `layout`, `title`, `date`, `author`, and topic metadata. Keep filenames lowercase and hyphenated: `2018-08-04-subexpressions.markdown`, `sql_equivalence_framework.png`. Use 2-space indentation in YAML and HTML where formatting is needed. Avoid renaming published asset paths unless all links are updated.

## Testing Guidelines
There is no automated test suite in this repository. Validate changes by running `bundle exec jekyll build` and checking for broken links, missing images, and Markdown rendering issues. For new posts, confirm the front matter parses correctly and the post appears with the expected permalink and title.

## Commit & Pull Request Guidelines
Recent history uses short imperative commits such as `Update index.md` and `Update research section with additional authors`. Follow that style: one focused change per commit, with the primary file or section named when useful. Pull requests should include a brief summary, note any changed pages or assets, and attach screenshots for visible layout changes. Link the relevant issue or context when updating publications, news, or profile details.

## Content Maintenance Tips
When editing `index.md`, preserve existing section order unless there is a clear reason to restructure it. Add new research, news, and talk entries in the established list format so diffs stay easy to review.
