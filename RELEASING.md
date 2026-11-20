# Releasing Atlas

Maintainer checklist for building and publishing a release. Follow it top to bottom.

## How the pieces fit

| Repo | What lives there |
|------|------------------|
| **A.T.L.A.S.** (this repo) | App source, `Dockerfile`, `.github/workflows/build.yml`. Pushing a `v*` tag here starts the CI builds. |
| **atlas-website** | The download buttons. `index.html` builds its download URLs from the `ATLAS_VERSION` constant and reads files from **atlas-website's own Releases**. |
| **A.T.L.A.S.-mobile** | Flutter source for the Android app. |

> **Important:** the CI `release` job publishes to *this* repo's Releases, but the website downloads from *atlas-website's* Releases. Any file the site needs has to be attached to the atlas-website release by hand (section 5).

## Versioning

One version number for every platform, for example `v1.4.7`.

- New feature: bump the minor (`v1.5.0`). Fix: bump the patch (`v1.4.8`).
- This repo and atlas-website have separate tag lists. Check this repo with `git tag --list 'v1.*'`, and use the same new number for the atlas-website release.
- Re-attach unchanged files (usually the APK) to every new release, and say so in the release notes.

## Quick checklist

- [ ] 1. Merge to `main` through a PR, pull, check dependencies
- [ ] 2. Build Linux with Docker, make the split zip, test
- [ ] 3. Push a tag to build Mac and Windows, download the artifacts
- [ ] 4. Android APK: rebuild, or reuse the last one
- [ ] 5. Create the release on atlas-website and attach every file
- [ ] 6. Update `ATLAS_VERSION` on the website (through a PR)
- [ ] 7. Verify the links and the live site

---

## 1. Prep

`main` is protected: no direct pushes and no force-pushes. Everything goes in through a PR.

```bash
git checkout main
git pull origin main
```

**Check that the build requirements are in sync.** Docker and CI install from `requirements-build.txt`, not `requirements.txt`, so a package missing there only fails at runtime (this is how `slowapi` and `qrcode` were missed before):

```bash
diff <(grep -v -E "nvidia-|^triton|^cuda-bindings|^cuda-pathfinder" requirements.txt | sort) <(sort requirements-build.txt)
```

No output means they match. A `<` line is a package missing from `requirements-build.txt`. A `>` line is in the build file but not in `requirements.txt`. Add or remove it in both files.

**Two other checks before building:**

- `config.yaml` is bundled into the build. Make sure it has no personal paths or settings in it.
- `.dockerignore` should exclude `.venv`, `Atlas-linux`, `dist`, `build` and `*.zip`. Otherwise `COPY . .` drags old builds into the image and slows every build.

## 2. Linux build (Docker)

```bash
cd ~/dev/A.T.L.A.S.
rm -rf Atlas-linux
docker rm -f atlas-temp 2>/dev/null
docker build -t atlas-builder .
docker create --name atlas-temp atlas-builder
docker cp atlas-temp:/app/dist/main ./Atlas-linux
docker rm atlas-temp
```

The build takes about 15 minutes. If the image build fails, nothing gets extracted, so don't use a stale `Atlas-linux` folder.

**Zip it, then test.** Zip first, because `install.sh` writes `Atlas.desktop` into the folder with paths from your machine. If that file gets zipped, downloaders get a launcher pointing at `/home/<you>/...`.

```bash
rm -f Atlas-linux-split.z*
zip -s 1900m -r Atlas-linux-split.zip Atlas-linux/
ls -lh Atlas-linux-split.*
```

`zip -s` can't update an existing archive, which is why the old parts are deleted first.

**Count the parts.** The website expects exactly three: `.z01`, `.z02` and `.zip`. If you get a different number, update the `files` array in `downloadAtlasLinux()` on the website (section 6).

**Test the folder:**

```bash
cd Atlas-linux
./install.sh
./main
```

Click through anything new in this release. Then delete the generated `Atlas.desktop` so it isn't left behind in the build folder.

## 3. Mac and Windows builds (GitHub Actions)

Pushing a tag starts the workflow, which builds Linux, Windows and macOS in parallel.

```bash
git tag --list 'v1.*'          # pick the next free number
git tag v1.4.7
git push origin v1.4.7         # name the remote; never use `git push --tags`
```

Watch the run on the repo's **Actions** tab. When it finishes, scroll to **Artifacts** at the bottom of the run's summary page. There is one zip each for `Atlas-linux`, `Atlas-windows` and `Atlas-mac`. Download the Mac and Windows ones.

### Known issues

- **Windows fails at checkout** with `Directory 'D:\a\A.T.L.A.S.\A.T.L.A.S.' does not exist`. The repo name ends in a period, which Windows strips from folder names. Nothing in the workflow gets around it. Use the temporary rename below.
- **The `release` job needs all three builds to pass.** If Windows fails, no release is created, but the artifacts are still on the run page.
- **The Linux zip is too big for a GitHub release.** It was about 5.5 GB when built locally, and GitHub caps release files at 2 GB, so the `release` job may fail on that upload. Ship Linux from the Docker split zip (section 2). Treat the CI Linux build as a smoke test.
- **macOS builds run on Apple Silicon** (arm64) runners, so the Mac zip is for M-series Macs only.
- **Mac and Windows builds are unsigned.** Expect OS security warnings on first launch, and mention it in the release notes.

### Building Windows: temporary repo rename

1. Make sure the Windows steps in `build.yml` use relative paths: `path: atlas` on checkout and `working-directory: atlas` on the other Windows steps. `actions/checkout` rejects absolute paths like `C:\atlas`.
2. Merge that through a PR first, and pull, so the rename window is short.
3. Rename the repo to a name with no trailing period, for example `ATLAS`: **Settings > General > Repository name**.
4. Push the tag (the old remote URL redirects):
   ```bash
   git tag v1.4.7
   git push origin v1.4.7
   ```
5. **Wait for the whole run to finish.** Don't rename back mid-run.
6. Rename the repo back to `A.T.L.A.S.`.
7. Confirm the old name works again: `git fetch origin`.

Don't create a new repo with either name while it's renamed, because that breaks the redirect.

## 4. Android APK

If the mobile app changed:

```bash
cd ~/dev/atlas_mobile
flutter build apk --release
cp build/app/outputs/flutter-apk/app-release.apk ~/Atlas.apk
```

If it didn't change, download `Atlas.apk` from the previous release and reuse it. The file must be named exactly `Atlas.apk`.

## 5. Create the release (atlas-website repo)

1. Go to atlas-website, then **Releases > Draft a new release**.
2. In **Choose a tag**, type the new version (for example `v1.4.7`) and pick **Create new tag on publish**. Leaving it blank gives "tag name can't be blank".
3. Set the title and paste the release notes (template below).
4. Attach every file. Keep these exact names, because the website builds its URLs from them:
   - `Atlas-linux-split.z01`
   - `Atlas-linux-split.z02`
   - `Atlas-linux-split.zip`
   - `Atlas.apk`
   - `Atlas-mac.zip` and `Atlas-windows.zip`, once those builds are working
5. Publish.

### Release notes template

```markdown
## What's new
- 

## Known limitations
- CPU-only build, so transcription is slower than on a GPU setup
- Requires Ollama installed locally with the `phi3:mini` model pulled
- Mac and Windows builds are unsigned, so your OS may show a security warning

## Downloads
- Linux: `Atlas-linux-split.z01` / `.z02` / `.zip` (see below to reassemble)
- Android: `Atlas.apk`, unchanged from vX.Y.Z, re-attached for consistency with this release's version tag
- macOS (Apple Silicon): `Atlas-mac.zip`
- Windows: coming soon

## Linux install
1. Download all 3 parts
2. Reassemble: `zip -F Atlas-linux-split.zip --out Atlas-linux-full.zip`
3. Unzip, then run `./install.sh` once to create a desktop launcher
4. Double-click `Atlas.desktop` to launch
```

Delete any line that doesn't apply to this release.

## 6. Update the website

In the atlas-website repo, in `index.html`:

1. Change the one version line:
   ```javascript
   const ATLAS_VERSION = 'v1.4.7';
   ```
   The footer text, the Android link and the Linux download all read from it.
2. If the Linux zip split into a different number of parts, update the `files` array in `downloadAtlasLinux()`.
3. Open a PR and merge it. GitHub Pages redeploys in a minute or two.

**When Mac or Windows are ready**, give each card's button an id and set its link from `RELEASE_BASE`, the same way the Android button works:

```html
<a href="#" id="macDownload" class="btn btn-ghost" style="align-self:flex-start;">Download</a>
```

```javascript
document.getElementById('macDownload').href = RELEASE_BASE + 'Atlas-mac.zip';
```

Also fix the card copy. The Mac card currently says `Atlas.dmg` and "Universal build, Intel and Apple Silicon", but CI produces `Atlas-mac.zip` for Apple Silicon only. The Windows card says `Atlas.exe`, but the file is `Atlas-windows.zip`.

## 7. Verify

Check that every file is on the release (replace the version). `200` or `206` means the file is there, and `404` means it's missing or misnamed:

```bash
VER=v1.4.7
BASE=https://github.com/sdoylelambda/atlas-website/releases/download/$VER
for f in Atlas.apk Atlas-linux-split.z01 Atlas-linux-split.z02 Atlas-linux-split.zip; do
  printf '%-26s' "$f"; curl -sL -r 0-0 -o /dev/null -w '%{http_code}\n' "$BASE/$f"
done
```

Then on the live site:

- The footer shows the new version
- The Android button downloads `Atlas.apk`
- The Linux button triggers all three parts

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `zip: cannot update a split archive` | Delete the old parts first: `rm -f Atlas-linux-split.z*` |
| `tag name can't be blank` on a release | Type the tag in **Choose a tag** and pick **Create new tag on publish** |
| Website download gives a 404 | The file is missing from that release, the name differs, or `ATLAS_VERSION` doesn't match a tag on atlas-website |
| A build ran on the wrong repo | `git remote -v`, and push tags with `git push origin <tag>` |
| Need to redo a tag | `git tag -d vX.Y.Z`, then `git push origin :refs/tags/vX.Y.Z`, then tag and push again |


